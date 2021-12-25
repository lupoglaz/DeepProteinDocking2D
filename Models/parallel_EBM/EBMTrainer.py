import torch
from torch import optim
import torch.nn as nn
import numpy as np

from DeepProteinDocking2D.Models import EQDockerGPU, EQScoringModel, EQRepresentation
from DeepProteinDocking2D.Models.Convolution import ProteinConv2D
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly

import random

from matplotlib import pylab as plt

class SampleBuffer:
	def __init__(self, num_samples, max_pos=100):
		self.num_samples = num_samples
		self.max_pos = max_pos
		self.buffer = {}
		for i in range(num_samples):
			self.buffer[i] = []
	
	def __len__(self, i):
		return len(self.buffer[i])
	
	def push(self, alphas, drs, index):
		alphas = alphas.detach().to(device='cpu')
		drs = drs.detach().to(device='cpu')

		for alpha, dr, idx in zip(alphas, drs, index):
			i = idx.item()
			self.buffer[i].append((alpha, dr))
			if len(self.buffer[i])>self.max_pos:
				self.buffer[i].pop(0)

	def get(self, index, num_samples=1, device='cuda'):
		alphas = []
		drs = []
		for idx in index:
			i = idx.item()
			if len(self.buffer[i])>=num_samples and random.randint(0,10)<7:
				lst = random.choices(self.buffer[i], k=num_samples)
				alpha = list(map(lambda x: x[0], lst))
				dr = list(map(lambda x: x[1], lst))
				alphas.append(torch.stack(alpha, dim=0))
				drs.append(torch.stack(dr, dim=0))
			else:
				alpha = torch.rand(num_samples, 1)*2*np.pi - np.pi
				dr = torch.rand(num_samples, 2)*50.0 - 25.0
				alphas.append(alpha)
				drs.append(dr)
		
		alphas = torch.stack(alphas, dim=0).to(device=device)
		drs = torch.stack(drs, dim=0).to(device=device)

		return alphas, drs


class EBMTrainer:
	def __init__(self, model, optimizer, num_buf_samples=10, device='cuda', num_samples=10, weight=1.0, step_size=10.0, sample_steps=100,
				global_step=True, add_positive=True, FI=False):
		self.model = model
		self.optimizer = optimizer
		self.buffer = SampleBuffer(num_buf_samples)
		self.buffer2 = SampleBuffer(num_buf_samples)
		self.global_step = global_step
		self.add_positive = add_positive

		self.num_samples = num_samples
		self.sample_steps = sample_steps
		self.weight = weight
		self.step_size = step_size
		self.device = device

		self.plot_idx = 0
		self.conv = ProteinConv2D()

		self.sig_dr = 0.05
		self.sig_alpha = 0.5
		self.docker = EQDockerGPU(EQScoringModel(EQRepresentation).to(device='cuda'))

		self.FI = FI
		if self.FI:
			##### load blank models and optimizers, once
			lr_interaction = 10 ** -3
			self.interaction_model = EBMInteractionModel().to(device=0)
			self.optimizer_interaction = optim.Adam(self.interaction_model.parameters(), lr=lr_interaction)

	def requires_grad(self, flag=True):
		parameters = self.model.parameters()
		for p in parameters:
			p.requires_grad = flag

	def load_checkpoint(self, path):
		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		checkpoint = torch.load(path)
		raw_model.load_state_dict(checkpoint)

	def dock_spatial(self, rec_repr, lig_repr):
		translations = self.conv(rec_repr, lig_repr)
		
		batch_size = translations.size(0)
		num_features = translations.size(1)
		L = translations.size(2)

		translations = translations.view(batch_size, num_features, L*L)
		translations = translations.transpose(1,2).contiguous().view(batch_size*L*L, num_features)
		scores = self.model.scorer(translations).squeeze()
		scores = scores.view(batch_size, L, L)
		
		minval_y, ind_y = torch.min(scores, dim=2, keepdim=False)
		minval_x, ind_x = torch.min(minval_y, dim=1)
		x = ind_x
		y = ind_y[torch.arange(batch_size), ind_x]
		
		x -= int(L/2)
		y -= int(L/2)
		
		# plt.imshow(scores[0,:,:].detach().cpu(), cmap='magma')
		# plt.plot([y[0].item()], [x[0].item()], 'xb')
		# plt.show()
		# sys.exit()
		return torch.stack([x,y], dim=1).to(dtype=lig_repr.dtype, device=lig_repr.device)
				
	def rotate(self, repr, angle):
		alpha = angle.detach()
		T0 = torch.cat([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
		T1 = torch.cat([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
		R = torch.stack([T0, T1], dim=1)
		curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
		return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

	def langevin(self, neg_alpha, neg_dr, rec_feat, lig_feat, neg_idx, temperature='cold'):
		noise_alpha = torch.zeros_like(neg_alpha)
		noise_dr = torch.zeros_like(neg_dr)

		self.requires_grad(False)
		self.model.eval()

		if self.global_step:
			with torch.no_grad():
				rlig_feat = self.rotate(lig_feat, neg_alpha)
				neg_dr = self.dock_spatial(rec_feat, rlig_feat)
		
		neg_alpha.requires_grad_()
		neg_dr.requires_grad_()
		langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

		if temperature == 'hot':
			self.sig_alpha = 0.5
			self.sig_dr *= 5

		lastN_neg_out = []

		for i in range(self.sample_steps):
			langevin_opt.zero_grad()

			pos_repr, _, A = self.model.mult(rec_feat, lig_feat, neg_alpha, neg_dr)
			neg_out = self.model.scorer(pos_repr)
			neg_out.mean().backward()

			langevin_opt.step()

			neg_dr.data += noise_dr.normal_(0, self.sig_dr)
			neg_alpha.data += noise_alpha.normal_(0, self.sig_alpha)

			neg_dr.data.clamp_(-rec_feat.size(2), rec_feat.size(2))
			neg_alpha.data.clamp_(-np.pi, np.pi)

			lastN_neg_out.append(neg_out.detach())

		if self.FI:
			return neg_alpha.detach(), neg_dr.detach(), lastN_neg_out
		else:
			return neg_alpha.detach(), neg_dr.detach()

	def step(self, data, epoch=None, train=True):
		receptor, ligand, translation, rotation, pos_idx = data
		
		pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_alpha = rotation.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_dr = translation.to(device=self.device, dtype=torch.float32)

		batch_size = pos_rec.size(0)
		num_features = pos_rec.size(1)
		L = pos_rec.size(2)
		
		neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=self.num_samples)
		neg_rec = pos_rec.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size*self.num_samples, num_features, L, L)
		neg_lig = pos_lig.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size*self.num_samples, num_features, L, L)
		neg_idx = pos_idx.unsqueeze(dim=1).repeat(1, self.num_samples).view(batch_size*self.num_samples)
		neg_alpha = neg_alpha.view(batch_size*self.num_samples, -1)
		neg_dr = neg_dr.view(batch_size*self.num_samples, -1)
		
		neg_rec_feat = self.model.repr(neg_rec).tensor
		neg_lig_feat = self.model.repr(neg_lig).tensor
		pos_rec_feat = self.model.repr(pos_rec).tensor
		pos_lig_feat = self.model.repr(pos_lig).tensor
		
		neg_alpha, neg_dr = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx)
		
		self.requires_grad(True)
		self.model.train()
		self.model.zero_grad()
		
		pos_out,_,_ = self.model.mult(pos_rec_feat, pos_lig_feat, pos_alpha, pos_dr)
		pos_out = self.model.scorer(pos_out)
		L_p = (pos_out + self.weight * pos_out ** 2).mean()
		neg_out,_,_ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr)
		neg_out = self.model.scorer(neg_out)
		L_n = (-neg_out + self.weight * neg_out ** 2).mean()
		loss = L_p + L_n
		loss.backward()

		print('L_p, L_n, \n', L_p, L_n)
		print('Loss\n', loss)

		if not train:
			with torch.no_grad():
				if pos_idx % 20 == 0:
					print('PLOTTING LOSS')
					filename = 'EBM_figs/IP_figs/IP_Loss_epoch' + str(epoch) + ' example number' + str(
						pos_idx.item())
					self.plot_IP_energy(L_p.detach().cpu().numpy(), L_n.detach().cpu().numpy(), epoch, pos_idx,
										filename)
					print('PLOTTING PREDICTION')
					filename = 'EBM_figs/IP_figs/IPpose_epoch' + str(epoch) + '_' + str(
						self.sample_steps) + 'samples_pose_after_LD' + str(pos_idx.item())
					self.plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD',
								   filename, pos_idx, epoch,
								   pos_alpha.squeeze().detach().cpu(), pos_dr.squeeze().detach().cpu())
		
		self.optimizer.step()
		if self.add_positive:
			self.buffer.push(pos_alpha, pos_dr, pos_idx)
		self.buffer.push(neg_alpha, neg_dr, neg_idx)
		
		return {"Loss": loss.item()}

	def step_parallel(self, data, epoch=None, train=True):
		gt_interact = None
		pos_alpha = None
		pos_dr = None
		if self.FI:
			receptor, ligand, gt_interact, pos_idx = data
			pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		else:
			receptor, ligand, translation, rotation, pos_idx = data
			pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			pos_alpha = rotation.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			pos_dr = translation.to(device=self.device, dtype=torch.float32)


		batch_size = pos_rec.size(0)
		num_features = pos_rec.size(1)
		L = pos_rec.size(2)

		# print(pos_idx, pos_idx.type(), pos_idx.shape)

		neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=self.num_samples)
		neg_alpha2, neg_dr2 = self.buffer2.get(pos_idx, num_samples=self.num_samples)

		neg_rec = pos_rec.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size * self.num_samples,
																					 num_features, L, L)
		neg_lig = pos_lig.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size * self.num_samples,
																					 num_features, L, L)
		neg_idx = pos_idx.unsqueeze(dim=1).repeat(1, self.num_samples).view(batch_size * self.num_samples)
		neg_alpha = neg_alpha.view(batch_size * self.num_samples, -1)
		neg_dr = neg_dr.view(batch_size * self.num_samples, -1)
		neg_alpha2 = neg_alpha2.view(batch_size * self.num_samples, -1)
		neg_dr2 = neg_dr2.view(batch_size * self.num_samples, -1)

		neg_rec_feat = self.model.repr(neg_rec).tensor
		neg_lig_feat = self.model.repr(neg_lig).tensor
		pos_rec_feat = self.model.repr(pos_rec).tensor
		pos_lig_feat = self.model.repr(pos_lig).tensor

		if self.FI:
			return self.interaction_prediction(neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr, neg_alpha2, neg_dr2,
											   pos_idx, neg_idx, receptor, ligand, gt_interact, epoch, train)
		else:
			return self.pose_prediction(pos_rec_feat, pos_lig_feat, neg_rec_feat, neg_lig_feat,
										pos_alpha, pos_dr, neg_alpha, neg_dr, neg_alpha2, neg_dr2,
										pos_idx, neg_idx, receptor, ligand, gt_interact, epoch, train)


	def pose_prediction(self, pos_rec_feat, pos_lig_feat, neg_rec_feat, neg_lig_feat, pos_alpha, pos_dr, neg_alpha, neg_dr, neg_alpha2, neg_dr2, pos_idx, neg_idx, receptor, ligand, gt_interact, epoch,  train):
		## parallel_EBM IP parallel model
		# print('\nTraining =', train)
		# print(epoch, '*' * 100)
		# with torch.no_grad():
		# 	translations = self.docker.dock_global(neg_rec_feat, neg_lig_feat)
		# 	scores = self.docker.score(translations)
		# 	score, rotation, translation = self.docker.get_conformation(scores)
		# 	neg_alpha = rotation.unsqueeze(0).unsqueeze(0).cuda()
		# 	neg_dr = translation.unsqueeze(0).cuda()

		neg_alpha, neg_dr = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx,
										  'cold')
		neg_alpha2, neg_dr2 = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx,
											'hot')

		# print('neg_alpha, neg_dr', neg_alpha, neg_dr)
		# print('neg_alpha, neg_dr grad', neg_alpha.grad, neg_dr.grad)

		self.requires_grad(True)
		self.model.train()
		self.model.zero_grad()
		pos_out, _, _ = self.model.mult(pos_rec_feat, pos_lig_feat, pos_alpha, pos_dr)
		pos_out = self.model.scorer(pos_out)
		L_p = (pos_out + self.weight * pos_out ** 2).mean()
		neg_out, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr)
		neg_out = self.model.scorer(neg_out)
		L_n = (-neg_out + self.weight * neg_out ** 2).mean()
		neg_out2, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha2, neg_dr2)
		neg_out2 = self.model.scorer(neg_out2)
		L_n2 = (-neg_out2 + self.weight * neg_out2 ** 2).mean()

		L_n = (L_n + L_n2) / 2
		loss = L_p + L_n
		loss.backward()

		plotting = True
		# plotting = False
		print(train)

		print('L_p, L_n, \n', L_p, L_n)
		print('Loss\n', loss)

		if not train:
			if plotting:
				with torch.no_grad():
					if pos_idx % 20 == 0:
						print('PLOTTING LOSS')
						filename = 'EBM_figs/IP_figs/IP_Loss_epoch' + str(epoch) + ' example number' + str(
							pos_idx.item())
						self.plot_IP_energy(L_p.detach().cpu().numpy(), L_n.detach().cpu().numpy(), epoch, pos_idx, filename)
						print('PLOTTING PREDICTION')
						filename = 'EBM_figs/IP_figs/IPpose_epoch' + str(epoch) + '_' + str(
							self.sample_steps) + 'samples_pose_after_LD' + str(pos_idx.item())
						self.plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD',
									   filename, pos_idx, epoch,
									   pos_alpha.squeeze().detach().cpu(), pos_dr.squeeze().detach().cpu())
		#
		# # if plotting and (pos_idx > 620 or pos_idx < 1):
		# # 	if epoch == 0:
		# 				print('PLOTTING INITIALIZATION')
		# 				filename = 'EBM_figs/IP_figs/epoch' + str(epoch) + '_' + str(
		# 					self.sample_steps) + 'samples_initialized_pose' + str(pos_idx.item())
		# 				self.plot_pose(receptor, ligand, rotation, translation, 'Initalization global step pose', filename,
		# 							   pos_idx, epoch)
		# 				print('PLOTTING PREDICTION')
		# 				filename = 'EBM_figs/IP_figs/epoch' + str(epoch) + '_' + str(
		# 					self.sample_steps) + 'samples_pose_after_LD' + str(pos_idx.item())
		# 				self.plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD', filename,
		# 									   pos_idx, epoch)

		self.optimizer.step()
		# never add postive for step parallel and 1D LD buffer
		if self.add_positive:
			# print('AP')
			self.buffer.push(pos_alpha, pos_dr, pos_idx)
			self.buffer2.push(pos_alpha, pos_dr, pos_idx)

		self.buffer.push(neg_alpha, neg_dr, neg_idx)
		self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)

		# return loss.item()
		return {"Loss": loss.item()}

	def interaction_prediction(self,neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr, neg_alpha2, neg_dr2, pos_idx, neg_idx, receptor, ligand, gt_interact,
							    epoch, train):
		plotting = True
		# plotting = False
		if epoch == 0:
			print(epoch, '*' * 100)
			with torch.no_grad():
				translations = self.docker.dock_global(neg_rec_feat, neg_lig_feat)
				scores = self.docker.score(translations)
				score, rotation, translation = self.docker.get_conformation(scores)
				neg_alpha = rotation.unsqueeze(0).unsqueeze(0).cuda()
				neg_dr = translation.unsqueeze(0).cuda()

		#### two sim, hot and cold
		# neg_alpha, neg_dr, lastN_E_cold = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(),
		#                                                 neg_lig_feat.detach(), neg_idx, sigma_dr=0.05, sigma_alpha=0.5)
		# neg_alpha, neg_dr, lastN_E_hot = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(),
		#                                                neg_lig_feat.detach(), neg_idx, sigma_dr=0.5, sigma_alpha=5)
		# pred_interact, deltaF = self.interaction_model(lastN_E_cold, lastN_E_hot)

		#### single sim
		neg_alpha, neg_dr, last100_E_samples = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(),
															 neg_lig_feat.detach(), neg_idx)
		pred_interact, deltaF = self.interaction_model(last100_E_samples)

		# if plotting and (pos_idx > 620 or pos_idx < 1):
		# 	if epoch == 0:
		# 		print('PLOTTING INITIALIZATION')
		# 		filename = 'EBM_figs/FI_figs/epoch' + str(epoch) + '_' + str(
		# 			self.sample_steps) + 'samples_initialized_pose' + str(pos_idx.item())
		# 		self.plot_pose(receptor, ligand, rotation, translation, 'Initalization global step pose', filename,
		# 					   pos_idx, epoch)
		# 		print('PLOTTING PREDICTION')
		# 		filename = 'EBM_figs/FI_figs/epoch' + str(epoch) + '_' + str(
		# 			self.sample_steps) + 'samples_pose_after_LD' + str(pos_idx.item())
		# 		self.plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD', filename,
		# 					   pos_idx, epoch)

		# print('*' * 100, 'example '+str(pos_idx.item()))
		# print('Initialized neg_alpha and neg_dr with minimum energy pose')
		# print('Energy, rotation, translation')
		# print(score, rotation, translation)
		# print('*' * 100, 'example '+str(pos_idx.item()))
		# print('AFTER LD')
		# print('Energy, rotation, translation')
		# print(last100_E_samples[-1], neg_alpha, neg_dr)

		if train:
			self.requires_grad(True)
			self.model.train()
			self.model.zero_grad()

			# if plotting and epoch == 0 and (pos_idx > 50 or pos_idx < 1):
			#     self.plot_feats(pos_rec _feat, pos_lig_feat, neg_rec_feat, neg_lig_feat)

			BCEloss = torch.nn.BCELoss()
			l1_loss = torch.nn.L1Loss()
			loss = BCEloss(pred_interact.squeeze(), gt_interact.squeeze().cuda())
			#
			# w = 10 ** -5
			# L_reg = w * l1_loss(deltaF.squeeze(), torch.zeros(1).squeeze().cuda())
			# loss = BCEloss(pred_interact.squeeze(), gt_interact.squeeze().cuda()) + L_reg

			loss.backward()
			print('\n PREDICTED', pred_interact.item(), '; GROUND TRUTH', gt_interact.item())
			self.optimizer.step()
			self.optimizer_interaction.step()
			self.buffer.push(neg_alpha, neg_dr, neg_idx)
			self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)
			return loss.item()

	def plot_feats(self, pos_rec_feat, pos_lig_feat, neg_rec_feat, neg_lig_feat):
		pos_rec_feat = pos_rec_feat.squeeze()
		pos_lig_feat = pos_lig_feat.squeeze()
		pos_rec_bulk, pos_rec_bound = pos_rec_feat[0, :, :], pos_rec_feat[1, :, :]
		pos_lig_bulk, pos_lig_bound = pos_lig_feat[0, :, :], pos_lig_feat[1, :, :]

		neg_rec_feat = neg_rec_feat.squeeze()
		neg_lig_feat = neg_lig_feat.squeeze()
		rec_bulk, rec_bound = neg_rec_feat[0, :, :], neg_rec_feat[1, :, :]
		lig_bulk, lig_bound = neg_lig_feat[0, :, :], neg_lig_feat[1, :, :]

		with torch.no_grad():
			pos_lig_bulk = pos_lig_bulk.detach().cpu() / pos_lig_bulk.detach().cpu().max()
			pos_lig_bound = pos_lig_bound.detach().cpu() / pos_lig_bound.detach().cpu().max()
			pos_rec_bulk = pos_rec_bulk.detach().cpu() / pos_rec_bulk.detach().cpu().max()
			pos_rec_bound = pos_rec_bound.detach().cpu() / pos_rec_bound.detach().cpu().max()

			neg_lig_bulk = lig_bulk.detach().cpu() / lig_bulk.detach().cpu().max()
			neg_lig_bound = lig_bound.detach().cpu() / lig_bound.detach().cpu().max()
			neg_rec_bulk = rec_bulk.detach().cpu() / rec_bulk.detach().cpu().max()
			neg_rec_bound = rec_bound.detach().cpu() / rec_bound.detach().cpu().max()

			lig_plot = np.hstack((neg_lig_bulk, neg_lig_bound))
			rec_plot = np.hstack((neg_rec_bulk, neg_rec_bound))
			pos_lig_plot = np.hstack((pos_lig_bulk, pos_lig_bound))
			pos_rec_plot = np.hstack((pos_rec_bulk, pos_rec_bound))
			pos_plot = np.vstack((pos_rec_plot, pos_lig_plot))
			neg_plot = np.vstack((rec_plot, lig_plot))

			image = np.vstack((pos_plot, neg_plot))
			# plt.imshow(image)
			plt.imshow(neg_plot)
			plt.colorbar()
			plt.title('Bulk', loc='left')
			plt.title('Boundary', loc='right')
			plt.show()

	def plot_pose(self, receptor, ligand, rotation, translation, plot_title, filename, pos_idx, epoch, gt_rot=0,
				  gt_txy=(0, 0)):
		pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(),
							 ligand.squeeze().detach().cpu().numpy(),
							 rotation.detach().cpu().numpy(),
							 (translation.squeeze()[0].detach().cpu().numpy(),
							  translation.squeeze()[1].detach().cpu().numpy()),
							 gt_rot,
							 gt_txy)
		if self.FI:
			plt.imshow(pair[100:, :].transpose())
		else:
			plt.imshow(pair[:, :].transpose())
		plt.title('EBM FI Input', loc='left')
		plt.title(plot_title, loc='right')
		plt.suptitle(filename)
		if pos_idx < 1 and epoch == 0:
			plt.savefig(filename)
			plt.show()
		else:
			plt.savefig(filename)
		plt.close()

	def plot_IP_energy(self, L_p, L_n, epoch, pos_idx, filename):
		print('L_p, L_n', L_p, L_n)
		f, ax = plt.subplots(figsize=(6, 6))

		axes_lim = (-0.25, 0.25)
		ax.scatter(L_n, L_p, c=".3")
		ax.plot(axes_lim, axes_lim, ls="--", c=".3")
		ax.set(xlim=axes_lim, ylim=axes_lim)
		ax.set_ylabel('L_p')
		ax.set_xlabel('L_n two temp simulation ')
		plt.quiver([0], [0], [L_n], [L_p], angles='xy', scale_units='xy', scale=1)
		plt.quiver([0], [L_p], color=['r'], angles='xy', scale_units='xy', scale=1)
		plt.quiver([L_n], [0], color=['b'], angles='xy', scale_units='xy', scale=1)
		plt.title(
			'IP Loss: Difference in L_p and L_n\n' + 'epoch ' + str(epoch) + ' example number' + str(pos_idx.item()))
		# plt.show()
		if pos_idx < 1 and epoch == 0:
			plt.savefig(filename)
			plt.show()
		else:
			plt.savefig(filename)
		plt.close()

class EBMInteractionModel(nn.Module):
	def __init__(self):
		super(EBMInteractionModel, self).__init__()

		self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

	def forward(self, sampling):
		E = torch.stack(sampling, dim=0)
		F = -torch.logsumexp(-E, dim=(0,1,2))

		deltaF = F - self.F_0
		pred_interact = torch.sigmoid(-deltaF)

		with torch.no_grad():
			print('F', F.item())
			print('\n(deltaF - F_0): ', deltaF.item())
			print('F_0: ', self.F_0.item(), 'F_0 grad', self.F_0.grad)

		return pred_interact.squeeze(), deltaF.squeeze()
