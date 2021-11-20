import torch
from torch import optim
import torch.nn as nn
import numpy as np

from Models import ProteinConv2D
from tqdm import tqdm
import random
from math import cos, sin

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

	def get(self, index, num_samples, device='cuda'):
		alphas = []
		drs = []
		for idx in index:
			i = idx.item()
			if len(self.buffer[i])>=num_samples and random.randint(0,10)<7:
				# print('if statement')
				# print('len buffer', len(self.buffer[i]))
				lst = random.choices(self.buffer[i], k=num_samples)
				alpha = list(map(lambda x: x[0], lst))
				dr = list(map(lambda x: x[1], lst))
				alphas.append(torch.stack(alpha, dim=0))
				drs.append(torch.stack(dr, dim=0))
			else:
				# print('else statement')
				# print('len buffer', len(self.buffer[i]), self.buffer[i])
				alpha = torch.rand(num_samples, 1)*2*np.pi - np.pi
				dr = torch.rand(num_samples, 2)*50.0 - 25.0
				alphas.append(alpha)
				drs.append(dr)
				# print('\nalpha', alpha)
				# print('dr', dr)

		alphas = torch.stack(alphas, dim=0).to(device=device)
		drs = torch.stack(drs, dim=0).to(device=device)

		return alphas, drs


class EBMTrainer:
	def __init__(self, model, optimizer, num_buf_samples=10, device='cuda', num_samples=10, weight=1.0, step_size=10.0, sample_steps=100,
				global_step=True, add_positive=True, sigma_dr=0.05, sigma_alpha=0.5, FI=False):
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

		self.FI = FI
		if self.FI:
			##### load blank models and optimizers, once
			lr_interaction = 10 ** 0
			self.interaction_model = EBMInteractionModel().to(device=0)
			self.optimizer_interaction = optim.Adam(self.interaction_model.parameters(), lr=lr_interaction)

	def requires_grad(self, flag=True):
		parameters = self.model.parameters()
		for p in parameters:
			p.requires_grad = flag
			# print('parameters',p, flag, p.grad)

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

	def langevin(self, neg_alpha, neg_dr, rec_feat, lig_feat, neg_idx, sigma_dr=0.5, sigma_alpha=5):
		noise_alpha = torch.zeros_like(neg_alpha)
		noise_dr = torch.zeros_like(neg_dr)

		self.requires_grad(False)
		self.model.eval()

		# print(sigma_dr,sigma_alpha)

		if self.global_step:
			# print('GS')
			with torch.no_grad():
				rlig_feat = self.rotate(lig_feat, neg_alpha)
				neg_dr = self.dock_spatial(rec_feat, rlig_feat)

		neg_alpha.requires_grad_()
		neg_dr.requires_grad_()
		langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

		last100_neg_out = []

		for k in range(self.sample_steps):
			langevin_opt.zero_grad()

			pos_repr, _, A = self.model.mult(rec_feat, lig_feat, neg_alpha, neg_dr)
			neg_out = self.model.scorer(pos_repr)
			# neg_out.clone().mean().backward()
			neg_out.mean().backward()


			langevin_opt.step()

			neg_dr.data += noise_dr.normal_(0, sigma_dr)
			neg_alpha.data += noise_alpha.normal_(0, sigma_alpha)

			neg_dr.data.clamp_(-rec_feat.size(2), rec_feat.size(2))
			neg_alpha.data.clamp_(-np.pi, np.pi)

			last100_neg_out.append(neg_out.detach())

		if self.FI:
			return neg_alpha.detach(), neg_dr.detach(), last100_neg_out
		else:
			return neg_alpha.detach(), neg_dr.detach()

	def step_parallel(self, data, epoch=None, train=True):
		if self.FI:
			receptor, ligand, gt_interact, pos_idx = data
			pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			# gt_interact = gt_interact.to(device=self.device, dtype=torch.float32).squeeze()
			# print(gt_interact.shape)
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
			if train:
				neg_alpha, neg_dr, last100_neg_out_cold = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx, sigma_dr=0.05, sigma_alpha=0.5)
				neg_alpha2, neg_dr2, last100_neg_out_hot = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx, sigma_dr=0.5, sigma_alpha=5)

				self.requires_grad(True)
				self.model.train()
				self.model.zero_grad()

				# print('neg_alpha, neg_dr', neg_alpha, neg_dr)
				# print('neg_alpha, neg_dr grad', neg_alpha.grad, neg_dr.grad)

				pred_interact, deltaF, coldF, hotF = self.interaction_model(last100_neg_out_cold, last100_neg_out_hot)

				# w1 = 10 ** 0
				# w2 = 10 ** 0

				neg_rec_feat = neg_rec_feat.relu()
				neg_lig_feat = neg_lig_feat.relu()

				neg_out, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr)
				neg_out = self.model.scorer(neg_out)
				L_n = (-neg_out + self.weight * neg_out ** 2).mean()
				neg_out2, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha2, neg_dr2)
				neg_out2 = self.model.scorer(neg_out2)
				L_n2 = (-neg_out2 + self.weight * neg_out2 ** 2).mean()
				print(L_n, L_n2)
				# print(L_n.grad, L_n2.grad)

				neg_rec_feat = neg_rec_feat.squeeze()
				neg_lig_feat = neg_lig_feat.squeeze()
				rec_bulk, rec_bound = neg_rec_feat[0,:,:], neg_rec_feat[1,:,:]
				lig_bulk, lig_bound = neg_lig_feat[0,:,:], neg_lig_feat[1,:,:]
				plotting = False
				# plotting = True
				if plotting:
					with torch.no_grad():
						lig_plot = np.hstack((rec_bulk.detach().cpu(), rec_bound.detach().cpu()))
						rec_plot = np.hstack((lig_bulk.detach().cpu(), lig_bound.detach().cpu()))
						image = np.vstack((rec_plot, lig_plot))
						plt.imshow(image)
						plt.colorbar()
						plt.show()


				# print(loss_feats)

				BCEloss = torch.nn.BCELoss()
				l1_loss = torch.nn.L1Loss()

				w = 10 ** -5

				loss_feats = l1_loss(L_n, L_n2)
				L_reg = l1_loss(deltaF.squeeze(), torch.zeros(1).squeeze().cuda())
				loss = BCEloss(pred_interact.squeeze(), gt_interact.squeeze().cuda()) + loss_feats + L_reg

				# loss_feats = w * (torch.sum(lig_bulk) * torch.sum(rec_bulk)) + (torch.sum(lig_bound) * torch.sum(lig_bound))


				# L_reg = w * l1_loss(deltaF.squeeze(), torch.zeros(1).squeeze().cuda())
				# loss = BCEloss(pred_interact.squeeze(), gt_interact.squeeze().cuda()) + L_reg + loss_feats

				loss.backward()
				print('\n PREDICTED', pred_interact.item(), '; GROUND TRUTH', gt_interact.item())
				self.optimizer.step()
				self.optimizer_interaction.step()
				self.buffer.push(neg_alpha, neg_dr, neg_idx)
				self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)
				return loss.item()
			else:
				self.model.eval()
				with torch.no_grad():
					neg_alpha, neg_dr, last100_neg_out = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(),
																	   neg_lig_feat.detach(), neg_idx, sigma_dr=0.05,
																	   sigma_alpha=0.5)
					neg_alpha2, neg_dr2, last100_neg_out2 = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(),
																	  neg_lig_feat.detach(), neg_idx, sigma_dr=0.5,
																	  sigma_alpha=5)

				# print('neg_alpha, neg_dr', neg_alpha, neg_dr)
				# print('neg_alpha, neg_dr grad', neg_alpha.grad, neg_dr.grad)

				last100_neg_out_cold = torch.stack(last100_neg_out, dim=0)
				last100_neg_out_hot = torch.stack(last100_neg_out2, dim=0)

				pred_interact, deltaF = self.interaction_model(last100_neg_out_cold, plotting=False)
				print('cold sim')
				pred_interact2, deltaF2 = self.interaction_model(last100_neg_out_hot, plotting=False)
				print('hot sim')

				mixed_deltaF = (deltaF + deltaF2) / 2
				pred_interact = torch.sigmoid(-mixed_deltaF)

				threshold = 0.5
				TP, FP, TN, FN = 0, 0, 0, 0
				p = pred_interact.item()
				a = gt_interact.item()
				if p >= threshold and a >= threshold:
					TP += 1
				elif p >= threshold and a < threshold:
					FP += 1
				elif p < threshold and a >= threshold:
					FN += 1
				elif p < threshold and a < threshold:
					TN += 1
				# print('returning', TP, FP, TN, FN)
				return TP, FP, TN, FN
		else:
			neg_alpha, neg_dr = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx, sigma_dr=0.05, sigma_alpha=0.5)
			neg_alpha2, neg_dr2 = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx, sigma_dr=0.5, sigma_alpha=5)

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
			loss = L_p + (L_n + L_n2)/2
			loss.backward()

			self.optimizer.step()
			# never add postive for step parallel and 1D LD buffer
			if self.add_positive:
				# print('AP')
				self.buffer.push(pos_alpha, pos_dr, pos_idx)
				self.buffer2.push(pos_alpha, pos_dr, pos_idx)

			self.buffer.push(neg_alpha, neg_dr, neg_idx)
			self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)

			return loss.item()

	def step(self, data, epoch=None):
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

		self.optimizer.step()
		if self.add_positive:
			# print('AP')
			self.buffer.push(pos_alpha, pos_dr, pos_idx)
		self.buffer.push(neg_alpha, neg_dr, neg_idx)

		return loss.item()


class EBMInteractionModel(nn.Module):
	def __init__(self):
		super(EBMInteractionModel, self).__init__()

		self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

	def forward(self, cold_sim, hot_sim):
		E_cold = torch.stack(cold_sim, dim=0)
		E_hot = torch.stack(hot_sim, dim=0)
		# print(E_cold.shape)
		coldF = -torch.logsumexp(-E_cold, dim=(0,1,2))
		hotF = -torch.logsumexp(-E_hot, dim=(0,1,2))
		print('coldF', coldF)
		print('hotF', hotF)

		sumF = coldF + hotF
		deltaF = sumF - self.F_0

		# deltaF = coldF + hotF + self.F_0
		# deltaF = coldF + hotF - self.F_0
		# deltaF = coldF - self.F_0
		pred_interact = torch.sigmoid(-deltaF)

		with torch.no_grad():
			print('\n(deltaF - F_0): ', deltaF.item())
			print('F_0: ', self.F_0.item(), 'F_0', self.F_0.grad)

		return pred_interact.squeeze(), deltaF.squeeze(), coldF, hotF
