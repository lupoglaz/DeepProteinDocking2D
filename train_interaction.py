import torch
from torch import optim
import argparse
from pathlib import Path

import numpy as np

from Models import CNNInteractionModel, EQScoringModel, EQInteraction
from Models import RankingLoss, EQRepresentation, EmptyBatch
from torchDataset import get_interaction_stream, get_interaction_stream_balanced
from tqdm import tqdm
import random

from SupervisedTrainer import SupervisedTrainer
from DockingTrainer import DockingTrainer

from DatasetGeneration import Protein, Complex

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")
import sys
sys.path.append('/home/sb1638/')

torch.cuda.empty_cache()


def test(stream, trainer, threshold=0.5):
	TP, FP, TN, FN = 0, 0, 0, 0
	for data in tqdm(stream):
		log_dict = trainer.eval(data, threshold=threshold)
		TP += log_dict["TP"]
		FP += log_dict["FP"]
		TN += log_dict["TN"]
		FN += log_dict["FN"]
	
	Accuracy = float(TP + TN)/float(TP + TN + FP + FN)
	if (TP+FP)>0:
		Precision = float(TP)/float(TP + FP)
	else:
		Precision = 0.0
	if (TP + FN)>0:
		Recall = float(TP)/float(TP + FN)
	else:
		Recall = 0.0
	F1 = Precision*Recall/(Precision + Recall+1E-5)
	MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1E-5)
	return Accuracy, Precision, Recall, F1, MCC

def test_threshold(stream, trainer, iter=0, logger=None):
	all_pred = []
	all_target = []

	if not (logger is None):
		cplxs = []
		repr = None
		max_num_log = 10
		num_log = 0
		
	for data in tqdm(stream):
		log_dict = trainer.eval(data)
		all_pred.append(log_dict["Pred"].clone())
		all_target.append(log_dict["Target"].clone())
		
		if not (logger is None) and num_log<max_num_log:
			receptor, ligand, target = data
			rec = Protein(receptor[0,:,:].clone().numpy())
			lig = Protein(ligand[0,:,:].clone().numpy())
			if num_log==0:
				repr = log_dict["Repr"].clone()
			rot = log_dict["Rotation"].clone().cpu().numpy()
			tr = log_dict["Translation"].squeeze().clone().cpu().numpy()
			cplxs.append(Complex(rec, lig, rot, tr))
			num_log += 1

	all_pred = torch.cat(all_pred, dim=0)
	all_target = torch.cat(all_target, dim=0)

	if not (logger is None):
		cell_size=100
		dock_image = np.zeros((cell_size, max_num_log*cell_size))
		for i,cplx in enumerate(cplxs):
			dock_image[:, cell_size*i:cell_size*(i+1)] = cplx.get_canvas(cell_size=cell_size)
		logger.add_image("DockFI/Model/Dock", dock_image, iter, dataformats='HW')

		repr_image = np.zeros((50, 100))
		repr_image[:,:50] = repr[0,:,:]
		repr_image[:,50:] = repr[1,:,:]
		logger.add_image("DockFI/Model/Repr", repr_image, iter, dataformats='HW')

		fig = plt.figure()
		pos_sel = (all_target==1)
		neg_sel = (all_target==0)
		plt.scatter(torch.arange(0, pos_sel.sum().item()), all_pred[pos_sel].cpu(), color='red')
		plt.scatter(torch.arange(0, neg_sel.sum().item()), all_pred[neg_sel].cpu(), color='blue')
		logger.add_figure("DockFI/Model/Scatter", fig, iter)

	sorted_pred, perm = torch.sort(all_pred)
	sorted_target = all_target[perm]
	target_true = (sorted_target == 1.0).to(dtype=torch.float32)
	target_false = (sorted_target == 0.0).to(dtype=torch.float32)
	
	cum_TP = torch.cumsum(target_true, dim=0)
	cum_FN = torch.cumsum(target_false, dim=0)

	cum_FP = torch.cumsum(target_true.flip(dims=(0,)), dim=0).flip(dims=(0,))
	cum_TN = torch.cumsum(target_false.flip(dims=(0,)), dim=0).flip(dims=(0,))
	
	acc = (cum_TP + cum_TN).to(dtype=torch.float)/(cum_TP+cum_TN+cum_FP+cum_FN).to(dtype=torch.float)
	Accuracy, idx = torch.max(acc, dim=0)
	TP = cum_TP[idx]
	FP = cum_FP[idx]
	TN = cum_TN[idx]
	FN = cum_FN[idx]
	threshold = sorted_pred[idx]
	if (TP+FP)>0:
		Precision = float(TP)/float(TP + FP)
	else:
		Precision = 0.0
	if (TP + FN)>0:
		Recall = float(TP)/float(TP + FN)
	else:
		Recall = 0.0
	F1 = Precision*Recall/(Precision + Recall+1E-5)
	MCC = (TP*TN - FP*FN)/torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1E-5).item()
	return Accuracy.item(), Precision, Recall, F1, MCC, threshold


def write_validloss(filename, epoch=None):
	log_format = '%f\t%f\t%f\t%f\t%f\n'
	log_header = 'Accuracy\tPrecision\tRecall\tF1score\tMCC\n'
	with open('Log/' + str(args.experiment) + '/'+filename, 'a') as fout:
		# fout.write('Epoch ' + str(check_epoch) + '\n')
		fout.write('Epoch' + str(epoch) + '\n')
		fout.write(log_header)
		fout.write(log_format % (Accuracy, Precision, Recall, F1, MCC))
	fout.close()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='Debug', type=str)
	parser.add_argument('-data_dir', default='Log', type=str)

	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')
	parser.add_argument('-test_epoch', default=0, type=int)

	parser.add_argument('-resnet', action='store_const', const=lambda:'resnet', dest='model')
	parser.add_argument('-docker', action='store_const', const=lambda:'docker', dest='model')

	parser.add_argument('-batch_size', default=8, type=int)
	parser.add_argument('-num_epochs', default=100, type=int)
	parser.add_argument('-pretrain', default=None, type=str)
	parser.add_argument('-gpu', default=1, type=int)

	args = parser.parse_args()

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			if i == args.gpu:
				print('->', i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
			else:
				print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(args.gpu)

	## shuffle=True does not run... ValueError: batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
	# train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=1000, shuffle=True)

	train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=1000, shuffle=False)
	train_small_stream = get_interaction_stream('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=50)
	# train_stream = train_small_stream
	valid_stream = get_interaction_stream('DatasetGeneration/interaction_data_valid.pkl', batch_size=args.batch_size, max_size=1000)
	test_stream = get_interaction_stream('DatasetGeneration/interaction_data_test.pkl', batch_size=args.batch_size, max_size=1000)

	logger = SummaryWriter(Path(args.data_dir)/Path(args.experiment))

	if args.model() == 'resnet':
		model = CNNInteractionModel().cuda()
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = SupervisedTrainer(model, optimizer, type='int')

	elif args.model() == 'docker':
		repr = EQRepresentation(bias=False)
		scoring_model = EQScoringModel(repr=repr).to(device='cuda')
		if not(args.pretrain is None):
			trainer = DockingTrainer(scoring_model, None, type='pos')
			trainer.load_checkpoint(Path('Log')/Path(args.pretrain)/Path('dock_ebm.th'))

		model = EQInteraction(scoring_model).cuda()
		optimizer = optim.Adam(scoring_model.parameters(), lr=1e-3)
		trainer = DockingTrainer(model, optimizer, type='int', omega=1.0)
	
	#TRAINING 
	if args.cmd() == 'train':
		iter = 0
		for epoch in range(args.num_epochs):
			losses = []
			if args.model() == 'resnet':
				for data in tqdm(train_stream):
					log_dict = trainer.step(data)
					logger.add_scalar("DockFI/Train/Loss", log_dict["Loss"], iter)
					iter += 1
					losses.append(log_dict["Loss"])

				print(f'Loss {np.mean(losses)}')
				threshold = 0.5

			elif args.model() == 'docker':
				for data in tqdm(train_stream):
					try:
						log_dict = trainer.step(data)
					except EmptyBatch:
						continue
					logger.add_scalar("DockFI/Train/Loss", log_dict["LossRanking"], iter)
					logger.add_scalar("DockFI/Train/ScoreMean", torch.mean(log_dict["P"]).item(), iter)
					logger.add_scalar("DockFI/Train/ScoreStd", torch.std(log_dict["P"]).item(), iter)
					logger.add_scalar("DockFI/Train/Missclass", log_dict["Loss"], iter)
					for i, param in enumerate(trainer.model.parameters()):
						if param.ndimension() > 0:
							logger.add_histogram(f'DockFI/Model/{i}', param.detach().cpu(), iter)

					iter += 1
					losses.append(log_dict["Loss"])
							
				print(f'Loss {np.mean(losses)}')
				Accuracy, Precision, Recall, F1, MCC, threshold = test_threshold(train_small_stream, trainer, iter, logger)
				print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1} MCC: {MCC}')


				logger.add_scalar("DockFI/Valid/acc", Accuracy, epoch)
				logger.add_scalar("DockFI/Valid/prec", Precision, epoch)
				logger.add_scalar("DockFI/Valid/rec", Recall, epoch)
				logger.add_scalar("DockFI/Valid/MCC", MCC, epoch)

				## write validation loss to file
				write_validloss('log_FI_train_smallsetAPR.txt', epoch=epoch)
			
			torch.save(model.state_dict(), Path(args.data_dir)/Path(args.experiment)/Path('model_epoch'+str(epoch)+'.th'))
		
	#TESTING
	elif args.cmd() == 'test':
		assert args.test_epoch
		trainer.load_checkpoint(Path(args.data_dir)/Path(args.experiment)/Path('model_epoch'+str(args.test_epoch)+'.th'))
		if args.model() == 'docker':
			iter = 0
			Accuracy, Precision, Recall, F1, MCC, threshold = test_threshold(train_small_stream, trainer, iter, logger)
			print(f'Threshold {threshold}')
			print(f'Threshold Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1} MCC: {MCC}')
		else:
			threshold = 0.5
		
		vAccuracy, vPrecision, vRecall, vF1, vMCC = test(valid_stream, trainer, threshold=threshold)
		print(f'Validation Acc: {vAccuracy} Prec: {vPrecision} Rec: {vRecall} F1: {vF1} MCC: {vMCC}')

		write_validloss('log_FI_validsetAPR.txt', epoch=None)

		tAccuracy, tPrecision, tRecall, tF1, tMCC = test(test_stream, trainer, threshold=threshold)
		print(f'Test Acc: {tAccuracy} Prec: {tPrecision} Rec: {tRecall} F1: {tF1} MCC: {tMCC}')

		write_validloss('log_FI_testsetAPR.txt', epoch=args.test_epoch)

	# logger.add_hparams(	{'ModelType': args.model(), 'Pretrain': args.pretrain},
		# 					{'hparam/valid_acc': vAccuracy, 'hparam/valid_prec': vPrecision, 'hparam/valid_rec': vRecall, 
		# 					'hparam/valid_F1': vF1, 'hparam/valid_MCC': vMCC,
		# 					'hparam/test_acc': tAccuracy, 'hparam/test_prec': tPrecision, 'hparam/test_rec': tRecall, 
		# 					'hparam/test_F1': tF1, 'hparam/test_MCC': tMCC,})
	else:
		raise(ValueError())

# python train_interaction.py -experiment BFInteraction_georgydefaults_check -docker -train -batch_size 8 -num_epochs 10
# $ python train_interaction.py -experiment GEORGY_BFinteraction_PRETRAIN_shuffleFalse -docker -train -batch_size 8 -num_epochs 10
