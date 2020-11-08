import torch
from torch import optim

import numpy as np

from Models import EQScoringModelV2, EQDockModel
from torchDataset import get_dataset_stream
from tqdm import tqdm
import random

from StochTrainer import StochTrainer
from DockTrainer import DockTrainer

from DatasetGeneration import Protein, Complex


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

def run_docking_model(data, docker, epoch=None):
	receptor, ligand, translation, rotation, indexes = data
	batch_size = rotation.size(0)
	receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	translation = translation.to(device='cuda', dtype=torch.float)
	rotation = rotation.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	docker.eval()
	pred_angles, pred_translations = docker(receptor, ligand)
	
	if not epoch is None:
		dict = {"translations": docker.top_translations[0,:,:].cpu(),
				"rotations": (docker.angles, docker.top_rotations),
				"receptors": receptor.cpu(),
				"ligands": ligand.cpu(),
				"rotation": rotation.cpu(),
				"translation": translation.cpu(),
				"pred_rotation": pred_angles.cpu(),
				"pred_translation": pred_translations.cpu(),
				}
		with open(f"Log/valid_{epoch}.th", "wb") as fout:
			torch.save( dict, fout)

	score_diff = 0.0
	for i in range(batch_size):
		rec = Protein(receptor[i,0,:,:].cpu().numpy())
		lig = Protein(ligand[i,0,:,:].cpu().numpy())
		angle = rotation[i].item()
		pos = translation[i,:].cpu().numpy()
		cplx_correct = Complex(rec, lig, angle, pos)
		score_correct = cplx_correct.score(boundary_size=3, weight_bulk=1.0)
		angle_pred = pred_angles[i].item()
		pos_pred = pred_translations[i,:].cpu().numpy()
		cplx_pred = Complex(rec, lig, angle_pred, pos_pred)
		score_pred = cplx_pred.score(boundary_size=3, weight_bulk=1.0)
		score_diff = (-score_correct + score_pred)

	# pred_angle_vec = torch.cat([torch.cos(pred_angles), torch.sin(pred_angles)], dim=1)
	# answ_angle_vec = torch.cat([torch.cos(rotation), torch.sin(rotation)], dim=1)
	# La = torch.sqrt((pred_angle_vec - answ_angle_vec).pow(2).sum(dim=1)).mean().item()
	# Lr = torch.sqrt((pred_translations - translation).pow(2).sum(dim=1)).mean().item()/50.0
	# return La + Lr
	return float(score_diff)/float(batch_size)


if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/dataset_valid.pkl', batch_size=1)
	valid_stream = get_dataset_stream('DatasetGeneration/dataset_valid.pkl', batch_size=1)
	
	model = EQScoringModelV2().to(device='cuda')
	# model.eval()
	# model.load_state_dict(torch.load('Log/dock_ebm.th'))
	
	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
	buffer = SampleBuffer(num_samples=len(train_stream)*64)

	trainer = StochTrainer(model, optimizer, buffer)
	# trainer = DockTrainer(model, optimizer, buffer)

	with open('Log/log_train_scoring_v2.txt', 'w') as fout:
		fout.write('Epoch\tLoss\n')
	with open('Log/log_valid_scoring_v2.txt', 'w') as fout:
		fout.write('Epoch\tLoss\n')
	
	losses_train = []
	losses_valid = []
	for epoch in range(100):
		loss = []
		for data in tqdm(train_stream):
			loss.append([trainer.step_stoch(data, epoch=epoch)])
			break
		
		av_loss = np.average(loss, axis=0)[0,:]
		
		print('Epoch', epoch, 'Train Loss:', av_loss)
		with open('Log/log_train_scoring_v2.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch,av_loss[0]))
		
		
		if (epoch+1)%10 == 0:
			model.eval()
			torch.save(model.state_dict(), 'Log/dock_ebm.th')

		loss = []
		docker = EQDockModel(model, num_angles=120)
		for data in tqdm(valid_stream):
			loss.append(run_docking_model(data, docker, epoch=epoch))
			break
		
		av_loss = np.average(loss, axis=0)
		print('Epoch', epoch, 'Valid Loss:', av_loss)
		with open('Log/log_valid_scoring_v2.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch, av_loss))		
		

	

			





