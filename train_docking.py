import torch
from torch import optim

import numpy as np

from Models import EQScoringModel, EQDockerGPU
from torchDataset import get_dataset_stream
from tqdm import tqdm
import random

from EBMTrainer import EBMTrainer

from DatasetGeneration import Protein, Complex

def run_docking_model(data, docker, epoch=None):
	receptor, ligand, translation, rotation, indexes = data
	receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	translation = translation.to(device='cuda', dtype=torch.float)
	rotation = rotation.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	docker.eval()
	pred_angles, pred_translations = docker(receptor, ligand)
	
	if not epoch is None:
		log_data = {"translations": docker.top_translations.cpu(),
				"rotations": (docker.angles.cpu(), docker.top_rotations),
				"receptors": receptor.cpu(),
				"ligands": ligand.cpu(),
				"rotation": rotation.cpu(),
				"translation": translation.cpu(),
				"pred_rotation": pred_angles.cpu(),
				"pred_translation": pred_translations.cpu(),
				}
		

	score_diff = 0.0
	rec = Protein(receptor[0,0,:,:].cpu().numpy())
	lig = Protein(ligand[0,0,:,:].cpu().numpy())
	angle = rotation[0].item()
	pos = translation[0,:].cpu().numpy()
	cplx_correct = Complex(rec, lig, angle, pos)
	score_correct = cplx_correct.score(boundary_size=3, a00=10.0, a11=0.4, a10=-1.0)
	angle_pred = pred_angles.item()
	pos_pred = pred_translations.cpu().numpy()
	cplx_pred = Complex(rec, lig, angle_pred, pos_pred)

	rmsd = lig.rmsd(pos, angle, pos_pred, angle_pred)
	score_pred = cplx_pred.score(boundary_size=3, a00=10.0, a11=0.4, a10=-1.0)
	score_diff = (-score_correct + score_pred)

	return float(rmsd), log_data


if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/docking_data_train.pkl', batch_size=32, max_size=100)
	valid_stream = get_dataset_stream('DatasetGeneration/docking_data_valid.pkl', batch_size=1, max_size=20)
	
	model = EQScoringModel().to(device='cuda')
	# model.eval()
	# model.load_state_dict(torch.load('Log/dock_ebm.th'))
	
	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
	trainer = EBMTrainer(model, optimizer, num_samples=10, num_buf_samples=len(train_stream)*64)
	
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
			# break
		
		av_loss = np.average(loss, axis=0)[0,:]
		
		print('Epoch', epoch, 'Train Loss:', av_loss)
		with open('Log/log_train_scoring_v2.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch,av_loss[0]))
		
		
		if (epoch+1)%10 == 0:
			torch.save(model.state_dict(), 'Log/dock_ebm.th')

		loss = []
		log_data = []
		docker = EQDockerGPU(model, num_angles=360)
		for data in tqdm(valid_stream):
			it_loss, it_log_data = run_docking_model(data, docker, epoch=epoch)
			loss.append(it_loss)
			log_data.append(it_log_data)
			# break

		with open(f"Log/valid_{epoch}.th", "wb") as fout:
			torch.save(log_data, fout)
		
		av_loss = np.average(loss, axis=0)
		print('Epoch', epoch, 'Valid Loss:', av_loss)
		with open('Log/log_valid_scoring_v2.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch, av_loss))