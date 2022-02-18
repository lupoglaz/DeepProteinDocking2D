import torch
from torch import optim
from pathlib import Path
import numpy as np
import argparse

from Models import EQScoringModel, EQDockerGPU, CNNInteractionModel, EQRepresentation
from torchDataset import get_docking_stream
from tqdm import tqdm
import random

from EBMTrainer import EBMTrainer
from SupervisedTrainer import SupervisedTrainer
from DockingTrainer import DockingTrainer

from DatasetGeneration import Protein, Complex
# from Logger import Logger
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

import sys
sys.path.append('/home/sb1638/')

def run_docking_model(data, docker, iter, logger=None):
	receptor, ligand, translation, rotation, indexes = data
	receptor = receptor.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	ligand = ligand.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	translation = translation.to(device='cuda', dtype=torch.float)
	rotation = rotation.to(device='cuda', dtype=torch.float).unsqueeze(dim=1)
	docker.eval()
	pred_angles, pred_translations = docker(receptor, ligand)
	
	rec = Protein(receptor[0,0,:,:].cpu().numpy())
	lig = Protein(ligand[0,0,:,:].cpu().numpy())
	angle = rotation[0].item()
	pos = translation[0,:].cpu().numpy()	
	angle_pred = pred_angles.item()
	pos_pred = pred_translations.cpu().numpy()
	rmsd = lig.rmsd(pos, angle, pos_pred, angle_pred)
	
	if not(logger is None):
		logger.add_image("DockIP/Docker/Translations", docker.top_translations, iter, dataformats='HW')
		fig = plt.figure()
		plt.plot(docker.angles.cpu(), docker.top_rotations)
		logger.add_figure("DockIP/Docker/Rotations", fig, iter)

		cell_size = 100
		plot_image = np.zeros((2*cell_size, cell_size))
		cplx = Complex(	Protein(receptor.squeeze().cpu().numpy()), 
						Protein(ligand.squeeze().cpu().numpy()), 
						rotation.cpu().item(), 
						translation.cpu().squeeze().numpy())
		plot_image[:cell_size, :] = cplx.get_canvas(cell_size=cell_size)
				
		cplx = Complex(Protein(receptor.squeeze().cpu().numpy()), 
						Protein(ligand.squeeze().cpu().numpy()), 
						angle_pred, 
						pos_pred)
		plot_image[cell_size:, :] = cplx.get_canvas(cell_size=cell_size)
		logger.add_image("DockIP/Docker/Dock", plot_image, iter, dataformats='HW')
		
	return float(rmsd)

def run_prediction_model(data, trainer, epoch=None):
	log_dict = trainer.eval(data)
	receptor, ligand, translation, rotation, _ = data
	log_data = {"receptors": receptor.unsqueeze(dim=1).cpu(),
				"ligands": ligand.unsqueeze(dim=1).cpu(),
				"rotation": rotation.squeeze().cpu(),
				"translation": translation.squeeze().cpu(),
				"pred_rotation": log_dict["Rotation"].squeeze().cpu(),
				"pred_translation": log_dict["Translation"].squeeze().cpu()}
	return log_dict["Loss"], log_data

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-data_dir', default='Log', type=str)
	parser.add_argument('-experiment', default='DebugDocking', type=str)
	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')

	parser.add_argument('-resnet', action='store_const', const=lambda:'resnet', dest='model')
	parser.add_argument('-ebm', action='store_const', const=lambda:'ebm', dest='model')
	parser.add_argument('-docker', action='store_const', const=lambda:'docker', dest='model')

	parser.add_argument('-gpu', default=1, type=int)
	parser.add_argument('-step_size', default=10.0, type=float)
	parser.add_argument('-num_samples', default=10, type=int)
	parser.add_argument('-batch_size', default=24, type=int)
	parser.add_argument('-num_epochs', default=100, type=int)

	parser.add_argument('-no_global_step', action='store_const', const=lambda:'no_global_step', dest='ablation')
	parser.add_argument('-no_pos_samples', action='store_const', const=lambda:'no_pos_samples', dest='ablation')
	parser.add_argument('-default', action='store_const', const=lambda:'default', dest='ablation')
	args = parser.parse_args()

	if (args.model is None):
		parser.print_help()
		sys.exit()

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(args.gpu)
	
	train_stream = get_docking_stream('DatasetGeneration/docking_data_train.pkl', batch_size=args.batch_size, max_size=None)
	valid_stream = get_docking_stream('DatasetGeneration/docking_data_valid.pkl', batch_size=1, max_size=None)
	
	if args.model() == 'resnet':
		model = CNNInteractionModel().to(device='cuda')
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = SupervisedTrainer(model, optimizer, type='pos')
	
	elif args.model() == 'ebm':
		repr = EQRepresentation(bias=False)
		model = EQScoringModel(repr=repr).to(device='cuda')
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		if args.ablation is None:
			print('My default')
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples, num_buf_samples=len(train_stream)*args.batch_size, step_size=args.step_size,
							global_step=True, add_positive=True)
		elif args.ablation() == 'no_global_step':
			print('No global step')
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples, num_buf_samples=len(train_stream)*args.batch_size, step_size=args.step_size,
							global_step=False, add_positive=True)
		elif args.ablation() == 'no_pos_samples':
			print('No positive samples')
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples, num_buf_samples=len(train_stream)*args.batch_size, step_size=args.step_size,
							global_step=True, add_positive=False)
		elif args.ablation() == 'default':
			print('Default')
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples, num_buf_samples=len(train_stream)*args.batch_size, step_size=args.step_size,
							global_step=False, add_positive=False)
	
	elif args.model() == 'docker':
		repr = EQRepresentation(bias=False)
		model = EQScoringModel(repr=repr).to(device='cuda')
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = DockingTrainer(model, optimizer, type='pos')
	
	
	### TRAINING
	if args.cmd() == 'train':
		logger = SummaryWriter(Path(args.data_dir)/Path(args.experiment))
		min_loss = float('+Inf')
		iter = 0
		for epoch in range(args.num_epochs):
			for data in tqdm(train_stream):
				log_dict = trainer.step(data, epoch=epoch)
				logger.add_scalar("DockIP/Loss/Train", log_dict["Loss"], iter)
				iter += 1

			loss = []
			log_data = []
			docker = EQDockerGPU(model, num_angles=360)
			for i, data in tqdm(enumerate(valid_stream)):
				if args.model() == 'resnet':
					it_loss, it_log_data = run_prediction_model(data, trainer, epoch=epoch)
				elif args.model() == 'ebm' or args.model() == 'docker':
					if i==0:
						it_loss = run_docking_model(data, docker, iter, logger)
					else:
						it_loss = run_docking_model(data, docker, iter)

				loss.append(it_loss)
						
			av_loss = np.average(loss, axis=0)
			logger.add_scalar("DockIP/Loss/Valid", av_loss, iter)

			print('Epoch', epoch, 'Valid Loss:', av_loss)
			if av_loss < min_loss:
				torch.save(model.state_dict(), Path('Log')/Path(args.experiment)/Path('model.th'))
				print(f'Model saved: min_loss = {av_loss} prev = {min_loss}')
				min_loss = av_loss
	
	### TESTING
	if args.cmd() == 'test':
		test_stream = get_docking_stream('DatasetGeneration/docking_data_test.pkl', batch_size=1, max_size=None)
		trainer.load_checkpoint(Path(args.data_dir)/Path(args.experiment)/Path('model.th'))
		docker = EQDockerGPU(model, num_angles=360)
		loss = []
		for data in tqdm(test_stream):
			if args.model() == 'resnet':
				it_loss, it_log_data = run_prediction_model(data, trainer, epoch=0)
			elif args.model() == 'ebm' or args.model() == 'docker':
				it_loss = run_docking_model(data, docker, 0)
			loss.append(it_loss)
		av_loss = np.average(loss, axis=0)
		print(f'Test result: {av_loss}')

		valid_stream = get_docking_stream('DatasetGeneration/docking_data_valid.pkl', batch_size=1, max_size=None)
		loss = []
		for data in tqdm(valid_stream):
			if args.model() == 'resnet':
				it_loss, it_log_data = run_prediction_model(data, trainer, epoch=0)
			elif args.model() == 'ebm' or args.model() == 'docker':
				it_loss = run_docking_model(data, docker, 0)
			loss.append(it_loss)
		av_loss = np.average(loss, axis=0)
		print(f'Valid result: {av_loss}')

		# if args.cmd == 'train':
		# 	ablation = "None"
		# 	if not(args.ablation is None): args.ablation()
		# 	logger.add_hparams(	{	'ModelType': args.model(), 
		# 							'Ablation': ablation, 
		# 							'StepSize': args.step_size, 'NumSamples': args.num_samples
		# 						}, 
		# 						{'hparam/valid_loss': min_loss, 'hparam/test_loss': av_loss})#, run_name=args.experiment)
