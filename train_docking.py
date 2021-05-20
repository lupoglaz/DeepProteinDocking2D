import torch
from torch import optim
from pathlib import Path
import numpy as np
import argparse

from Models import EQScoringModel, EQDockerGPU, CNNInteractionModel
from torchDataset import get_docking_stream
from tqdm import tqdm
import random

from EBMTrainer import EBMTrainer
from SupervisedTrainer import SupervisedTrainer

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
		

	rec = Protein(receptor[0,0,:,:].cpu().numpy())
	lig = Protein(ligand[0,0,:,:].cpu().numpy())
	angle = rotation[0].item()
	pos = translation[0,:].cpu().numpy()	
	angle_pred = pred_angles.item()
	pos_pred = pred_translations.cpu().numpy()
	rmsd = lig.rmsd(pos, angle, pos_pred, angle_pred)
	
	return float(rmsd), log_data

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='DebugCNN', type=str)
	parser.add_argument('-supervised', default=True, type=bool)
	args = parser.parse_args()

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_docking_stream('DatasetGeneration/docking_data_train.pkl', batch_size=24, max_size=None)
	valid_stream = get_docking_stream('DatasetGeneration/docking_data_valid.pkl', batch_size=1, max_size=None)
		
	if args.supervised:
		model = CNNInteractionModel(type='pos').to(device='cuda')
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = SupervisedTrainer(model, optimizer)
	else:
		model = EQScoringModel().to(device='cuda')
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = EBMTrainer(model, optimizer, num_samples=10, num_buf_samples=len(train_stream)*64, step_size=10.0)
	
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
			if args.supervised:
				it_loss, pred_trans, pred_rot = trainer.eval(data)
				receptor, ligand, translation, rotation, _ = data
				it_log_data = {	"receptors": receptor.unsqueeze(dim=1).cpu(),
							"ligands": ligand.unsqueeze(dim=1).cpu(),
							"rotation": rotation.squeeze().cpu(),
							"translation": translation.squeeze().cpu(),
							"pred_rotation": pred_rot.squeeze().cpu(),
							"pred_translation": pred_trans.squeeze().cpu()}
			else:
				it_loss, it_log_data = run_docking_model(data, docker, epoch=epoch)

			loss.append(it_loss)
			log_data.append(it_log_data)
		
		av_loss = np.average(loss, axis=0)
		logger.log_valid(av_loss)
		logger.log_data(log_data)

		print('Epoch', epoch, 'Valid Loss:', av_loss)
		if av_loss < min_loss:
			torch.save(model.state_dict(), logger.log_dir/Path('dock_ebm.th'))
			print(f'Model saved: min_loss = {av_loss} prev = {min_loss}')
			min_loss = av_loss