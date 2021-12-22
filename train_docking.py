import sys
sys.path.append('/home/sb1638/')

import torch
from torch import optim
from pathlib import Path
import numpy as np
import argparse

from Models import EQScoringModel, EQDockerGPU, CNNInteractionModel
from torchDataset import get_docking_stream, get_interaction_stream_balanced
from tqdm import tqdm
import random

from EBMTrainer import EBMTrainer
from SupervisedTrainer import SupervisedTrainer
from DockingTrainer import DockingTrainer

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

def run_prediction_model(data, trainer, epoch=None):
	loss, pred_trans, pred_rot = trainer.eval(data)
	receptor, ligand, translation, rotation, _ = data
	log_data = {"receptors": receptor.unsqueeze(dim=1).cpu(),
				"ligands": ligand.unsqueeze(dim=1).cpu(),
				"rotation": rotation.squeeze().cpu(),
				"translation": translation.squeeze().cpu(),
				"pred_rotation": pred_rot.squeeze().cpu(),
				"pred_translation": pred_trans.squeeze().cpu()}
	return loss, log_data


if __name__=='__main__':
	#### initialization torch settings
	# random_seed = 42
	# np.random.seed(random_seed)
	# torch.manual_seed(random_seed)
	# random.seed(random_seed)
	# torch.cuda.manual_seed(random_seed)
	# torch.backends.cudnn.deterministic = True
	# torch.cuda.set_device(0)

	# torch.autograd.set_detect_anomaly(True)
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
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
	parser.add_argument('-parallel', action='store_const', const=lambda:'parallel', dest='ablation')
	parser.add_argument('-parallel_noGSAP', action='store_const', const=lambda:'parallel_noGSAP', dest='ablation')
	parser.add_argument('-FI', action='store_const', const=lambda:'FI', dest='ablation')

	args = parser.parse_args()

	if (args.cmd is None) or (args.model is None):
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
		model = EQScoringModel().to(device='cuda')
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
		elif args.ablation() == 'parallel':
			print('Parallel two different distribution sigmas, with global step, no positive samples')
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
								 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
								 global_step=True, add_positive=False)
		elif args.ablation() == 'parallel_noGSAP':
			print('Parallel, two different distribution sigmas, no GS, no AP')
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
								 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
								 global_step=False, add_positive=False)
		elif args.ablation() == 'FI':
			print('Fact of interaction: using parallel, different distribution sigmas, no GS, no AP')
			max_size=25
			train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=max_size)
			valid_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_valid.pkl', batch_size=1, max_size=max_size//2)
			trainer = EBMTrainer(model, optimizer, num_samples=args.num_samples,
								 num_buf_samples=len(train_stream) * args.batch_size, step_size=args.step_size,
								 global_step=False, add_positive=False, FI=True)

	elif args.model() == 'docker':
		model = EQScoringModel().to(device='cuda')
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = DockingTrainer(model, optimizer, type='pos')


	if args.cmd() == 'train':
		logger = Logger.new(Path('Log')/Path(args.experiment))
		min_loss = float('+Inf')
		# pretrain = True
		pretrain = False
		if pretrain:
			print('*'*100)
			print('Loading pretrained model for EBM')
			trainer.load_checkpoint(Path('Log') / Path('pretrain_EBM_IP_100samples_1dLD') / Path('dock_ebm.th'))
			print(Path('Log') / Path('pretrain_EBM_IP_100samples_1dLD') / Path('dock_ebm.th'))
		for epoch in range(args.num_epochs):
			pos_idx = 0
			for data in tqdm(train_stream):
				if args.model() == 'ebm' and args.ablation and 'parallel' in args.ablation():
					print('\nrunning parallel')
					loss = trainer.step_parallel(data, epoch=epoch)
				elif args.ablation() == 'FI':
					print('\nFI parallel')
					receptor, ligand, gt_interact = data
					data = (receptor, ligand, gt_interact, torch.tensor(pos_idx).unsqueeze(0).cuda())
					loss = trainer.step_parallel(data, epoch=epoch)
					pos_idx += 1
				else:
					print('\nNOT PARALLEL')
					loss = trainer.step(data, epoch=epoch)
			if not args.ablation() == 'FI':
				logger.log_train(loss)

			if args.ablation() == 'FI':
				log_format = '%f\t%f\t%f\t%f\t%f\n'
				log_header = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
				TP, FP, TN, FN = 0, 0, 0, 0

				pos_idx = 0
				for data in tqdm(valid_stream):
					receptor, ligand, gt_interact = data
					data = (receptor, ligand, gt_interact, torch.tensor(pos_idx).unsqueeze(0).cuda())
					tp, fp, tn, fn = trainer.step_parallel(data, train=False)
					# print(tp, fp, tn,fn)
					TP += tp
					FP += fp
					TN += tn
					FN += fn
					pos_idx += 1


				Accuracy = float(TP + TN) / float(TP + TN + FP + FN)
				if (TP + FP) > 0:
					Precision = float(TP) / float(TP + FP)
				else:
					Precision = 0.0
				if (TP + FN) > 0:
					Recall = float(TP) / float(TP + FN)
				else:
					Recall = 0.0
				F1score = TP / (TP + 0.5 * (FP + FN) + 1E-5)

				MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1E-5)

				print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1score} MCC: {MCC}')

				with open('Log/'+str(args.experiment)+'/log_EBM_FI_validAPR.txt', 'a') as fout:
					# fout.write('Epoch ' + str(check_epoch) + '\n')
					fout.write('Epoch' + str(epoch) + '\n')
					fout.write(log_header)
					fout.write(log_format % (Accuracy, Precision, Recall, F1score, MCC))
				fout.close()
			else:
				loss = []
				log_data = []
				docker = EQDockerGPU(model, num_angles=360)
				for data in tqdm(valid_stream):
					if args.model() == 'resnet':
						it_loss, it_log_data = run_prediction_model(data, trainer, epoch=0)
					elif args.model() == 'ebm':
						it_loss, it_log_data = run_docking_model(data, docker, epoch=epoch)
					elif args.model() == 'docker':
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

	elif args.cmd() == 'test':
		trainer.load_checkpoint(Path('Log')/Path(args.experiment)/Path('dock_ebm.th'))
		if args.ablation() == 'FI':
			test_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_test.pkl', batch_size=1, max_size=None)
		else:
			test_stream = get_docking_stream('DatasetGeneration/docking_data_test.pkl', batch_size=1, max_size=None)
		logger = Logger.new(Path('Log')/Path(args.experiment)/Path('Test'))
		
		loss = []
		log_data = []
		docker = EQDockerGPU(model, num_angles=360)
		for data in tqdm(test_stream):
			if args.model() == 'resnet':
				it_loss, it_log_data = run_prediction_model(data, trainer, epoch=0)
			elif args.model() == 'ebm':
				it_loss, it_log_data = run_docking_model(data, docker, epoch=0)
			elif args.model() == 'docker':
				it_loss, it_log_data = run_docking_model(data, docker, epoch=0)

			loss.append(it_loss)
			log_data.append(it_log_data)
		
		av_loss = np.average(loss, axis=0)
		logger.log_data(log_data)

		print(f'Test result: {av_loss}')