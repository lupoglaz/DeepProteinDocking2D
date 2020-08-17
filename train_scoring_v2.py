import torch
from torch import optim

import numpy as np

from Models import EQScoringModelV2, EQDockModel
from torchDataset import get_dataset_stream
from tqdm import tqdm
import random

from matplotlib import pylab as plt
from celluloid import Camera
from DatasetGeneration import rotate_ligand

class SampleBuffer:
	def __init__(self, max_samples=1000):
		self.max_samples = max_samples
		self.buffer = []
	
	def __len__(self):
		return len(self.buffer)
	
	def push(self, receptors, ligands, alphas, drs):
		receptors = receptors.detach().to(device='cpu')
		ligands = ligands.detach().to(device='cpu')
		alphas = alphas.detach().to(device='cpu')
		drs = drs.detach().to(device='cpu')

		for rec, lig, alpha, dr in zip(receptors, ligands, alphas, drs):
			self.buffer.append((rec, lig, alpha, dr))
			if len(self)>self.max_samples:
				self.buffer.pop(0)
	
	def get(self, n_samples, device='cuda'):
		items = random.choices(self.buffer, k=n_samples)
		receptors, ligands, alphas, drs = zip(*items)
		receptors = torch.stack(receptors, dim=0).to(device=device)
		ligands = torch.stack(ligands, dim=0).to(device=device)
		alphas = torch.stack(alphas, dim=0).to(device=device)
		drs = torch.stack(drs, dim=0).to(device=device)
		return receptors, ligands, alphas, drs

def sample_buffer(buffer, receptor, ligand, batch_size=128, p=0.95, device='cuda'):
	if len(buffer) < 1:
		return (
			receptor,
			ligand,
			torch.rand(batch_size, 1, device=device),
			torch.rand(batch_size, 2, device=device)
		)

	n_replay = (np.random.rand(batch_size) < p).sum()

	replay_rec, replay_lig, replay_alpha, replay_dr = buffer.get(n_replay)
	random_rec = receptor[0:batch_size-n_replay,:,:]
	random_lig = ligand[0:batch_size-n_replay,:,:]
	random_alpha = torch.rand(batch_size - n_replay, 1, device=device)
	random_dr = torch.rand(batch_size - n_replay, 2, device=device)
	
	return (
		torch.cat([replay_rec, random_rec], 0),
		torch.cat([replay_lig, random_lig], 0),
		torch.cat([replay_alpha, random_alpha], 0),
		torch.cat([replay_dr, random_dr], 0),
	)

def requires_grad(parameters, flag=True):
	for p in parameters:
		p.requires_grad = flag

def clip_grad(parameters, optimizer):
	with torch.no_grad():
		for group in optimizer.param_groups:
			for p in group['params']:
				state = optimizer.state[p]

				if 'step' not in state or state['step'] < 1:
					continue

				step = state['step']
				exp_avg_sq = state['exp_avg_sq']
				_, beta2 = group['betas']

				bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
				p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def plot(rec, lig, rot, trans):
	box_size = int(rec.size(1))
	field_size = box_size*2
	field = np.zeros( (field_size, field_size) )

	rot_lig = rotate_ligand(lig[:,:].numpy(), rot[0].item()*180.0/np.pi)
	dx = int(trans[0].item()*50.0-25.0)
	dy = int(trans[1].item()*50.0-25.0)

	field[ 	int(field_size/2 - box_size/2): int(field_size/2 + box_size/2),
				int(field_size/2 - box_size/2): int(field_size/2 + box_size/2)] += rec.numpy()
	
	field[  int(field_size/2 - box_size/2 + dx): int(field_size/2 + box_size/2 + dx),
			int(field_size/2 - box_size/2 + dy): int(field_size/2 + box_size/2 + dy) ] += 2*rot_lig

	plt.imshow(field)
	

def step_stoch(data, model, optimizer, buffer, device='cuda', alpha=1, step_size=10.0, sample_step=100, animate=False):
	receptor, ligand, translation, rotation = data
	batch_size = receptor.size(0)

	pos_rec = receptor.to(device=device, dtype=torch.float32)
	pos_lig = ligand.to(device=device, dtype=torch.float32)
	pos_alpha = rotation.to(device, dtype=torch.float32).unsqueeze(dim=1)*(np.pi/180.0)
	pos_dr = translation.to(device, dtype=torch.float32)/50.0 + 0.5
			
	noise_alpha = torch.randn(batch_size, 1, device=device, dtype=torch.float32)
	noise_dr = torch.randn(batch_size, 2, device=device, dtype=torch.float32)

	neg_rec, neg_lig, neg_alpha, neg_dr = sample_buffer(buffer, pos_rec, pos_lig, batch_size)
	neg_dr.requires_grad_()
	neg_alpha.requires_grad_()
	
	requires_grad(model.parameters(), False)
	model.eval()
	
	if animate:
		fig = plt.figure(figsize=(12,6))
		camera = Camera(fig)

	for k in range(sample_step):
		if batch_size != neg_dr.shape[0]:
			noise_dr = torch.randn(neg_dr.shape[0], 2, device=device)
			noise_alpha = torch.randn(neg_dr.shape[0], 1, device=device)

		noise_dr.normal_(0, 0.001)
		noise_alpha.normal_(0, 0.001)
		neg_dr.data.add_(noise_dr.data)
		neg_alpha.data.add_(noise_alpha.data)

		neg_out = model(neg_rec, neg_lig, neg_alpha, neg_dr)
		neg_out.sum().backward()
		if animate:
			plot(neg_rec[0,:,:].detach().cpu(), neg_lig[0,:,:].detach().cpu(), neg_alpha[0,:].detach().cpu(), neg_dr[0,:].detach().cpu())
			camera.snap()
		
		# neg_dr.grad.data.clamp_(-0.05, 0.05)
		# neg_alpha.grad.data.clamp_(-0.05, 0.05)

		neg_dr.data.add_(-step_size, neg_dr.grad.data)
		neg_alpha.data.add_(-step_size, neg_alpha.grad.data)

		neg_dr.grad.detach_()
		neg_alpha.grad.detach_()
		neg_dr.grad.zero_()
		neg_alpha.grad.zero_()

		neg_dr.data.clamp_(0, 1)
		neg_alpha.data.clamp_(-np.pi, np.pi)
	
	if animate:
		animation = camera.animate()
		animation.save('animation.mp4')

	requires_grad(model.parameters(), True)
	model.train()
	model.zero_grad()
	
	pos_out = model(pos_rec, pos_lig, pos_alpha, pos_dr)
	neg_out = model(neg_rec, neg_lig, neg_alpha, neg_dr)

	loss = alpha * (pos_out ** 2 + neg_out ** 2)
	loss = loss + (pos_out - neg_out)
	loss = loss.mean()
	loss.backward()

	# clip_grad(parameters, optimizer)

	optimizer.step()

	buffer.push(neg_rec, neg_lig, neg_alpha, neg_dr)
	
	return (pos_out - neg_out).mean().item(),

def step_determ(data, model, optimizer, device='cuda', alpha=0.1):
	receptor, ligand, translation, rotation = data
	batch_size = receptor.size(0)

	pos_rec = receptor.to(device=device, dtype=torch.float32).unsqueeze(dim=1)
	pos_lig = ligand.to(device=device, dtype=torch.float32).unsqueeze(dim=1)
	pos_alpha = rotation.to(device, dtype=torch.float32).unsqueeze(dim=1)*(np.pi/180.0)
	pos_dr = translation.to(device, dtype=torch.float32)
	
	model.eval()
	
	docker = EQDockModel(model, num_angles=20)
	neg_alpha, neg_dr = docker(pos_rec, pos_lig)
	
	model.train()
	model.zero_grad()
		
	pos_out = model(pos_rec, pos_lig, pos_alpha, pos_dr)
	neg_out = model(pos_rec, pos_lig, neg_alpha, neg_dr)

	loss = alpha * (pos_out ** 2 + neg_out ** 2)
	loss = loss + (pos_out - neg_out)
	loss = loss.mean()
	loss.backward()

	# clip_grad(parameters, optimizer)

	optimizer.step()
	
	return (pos_out - neg_out).mean().item(),

def run_docking_model(data, docker):
	receptor, ligand, translation, rotation = data
	batch_size = rotation.size(0)
	receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	translation = translation.to(device='cuda', dtype=torch.float)
	rotation = rotation.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	docker.eval()
	pred_angles, pred_translations = docker(receptor, ligand)
	
	rotation = rotation * (np.pi/180.0)
	pred_angle_vec = torch.cat([torch.cos(pred_angles), torch.sin(pred_angles)], dim=1)
	answ_angle_vec = torch.cat([torch.cos(rotation), torch.sin(rotation)], dim=1)
	La = torch.sqrt((pred_angle_vec - answ_angle_vec).pow(2).sum(dim=1)).mean().item()
	Lr = torch.sqrt((pred_translations - translation).pow(2).sum(dim=1)).mean().item()/50.0
	return La, Lr


if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/toy_dataset_1000.pkl', batch_size=16)
	valid_stream = get_dataset_stream('DatasetGeneration/toy_dataset.pkl', batch_size=10)
	
	model = EQScoringModelV2().to(device='cuda')
	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
	buffer = SampleBuffer()

	with open('Log/log_train_scoring_v2.txt', 'w') as fout:
		fout.write('Epoch\tLoss\n')
	with open('Log/log_valid_scoring_v2.txt', 'w') as fout:
		fout.write('Epoch\tLossAngle\tLossR\n')
	
	for epoch in range(300):
		loss = []
		for data in tqdm(train_stream):
			loss.append([step_determ(data, model, optimizer)])
		
		av_loss = np.average(loss, axis=0)[0,:]
		print('Epoch', epoch, 'Train Loss:', av_loss)
		with open('Log/log_train_scoring_v2.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch,av_loss[0]))

		loss = []
		docker = EQDockModel(model, num_angles=60)
		for data in tqdm(valid_stream):
			loss.append(run_docking_model(data, docker))
		
		av_loss = np.average(loss, axis=0)
		print('Epoch', epoch, 'Valid Loss:', av_loss)
		with open('Log/log_train_scoring_v2.txt', 'a') as fout:
			fout.write('%d\t%f\t%f\n'%(epoch,av_loss[0], av_loss[1]))

	torch.save(model.state_dict(), 'Log/dock_scoring_v2.th') 
		

	

			





