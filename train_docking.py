import torch
from torch import optim
from pathlib import Path
import numpy as np
import argparse

from Models import EQScoringModel, EQDockerGPU
from torchDataset import get_docking_stream
from tqdm import tqdm
import random

from EBMTrainer import EBMTrainer

from DatasetGeneration import Protein, Complex
from Logger import Logger

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
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='Debug', type=str)
	args = parser.parse_args()

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_docking_stream('DatasetGeneration/docking_data_train.pkl', batch_size=24, max_size=100)
	valid_stream = get_docking_stream('DatasetGeneration/docking_data_valid.pkl', batch_size=1, max_size=30)
	
	model = EQScoringModel().to(device='cuda')
	
	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
	trainer = EBMTrainer(model, optimizer, num_samples=10, num_buf_samples=len(train_stream)*64, step_size=5.0)
	logger = Logger.new(Path('Log')/Path(args.experiment))
	
	min_loss = float('+Inf')
	for epoch in range(100):
		for data in tqdm(train_stream):
			loss = trainer.step_stoch(data, epoch=epoch)
			logger.log_train(loss)

		loss = []
		log_data = []
		docker = EQDockerGPU(model, num_angles=360)
		for data in tqdm(valid_stream):
			it_loss, it_log_data = run_docking_model(data, docker, epoch=epoch)
			loss.append(it_loss)
			log_data.append(it_log_data)
		
		av_loss = np.average(loss, axis=0)
		logger.log_valid(av_loss)
		logger.log_data(log_data)

		print('Epoch', epoch, 'Valid Loss:', av_loss)
		if av_loss < min_loss:
			torch.save(model.state_dict(), LOG_DIR/Path('dock_ebm.th'))
			print(f'Model saved: min_loss = {av_loss} prev = {min_loss}')
			min_loss = av_loss