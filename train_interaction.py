import torch
from torch import optim
import argparse
from pathlib import Path

import numpy as np

from Models import CNNInteractionModel, EQScoringModel, EQInteraction, SidInteraction, EQInteractionF, SharpLoss, RankingLoss, EQRepresentationSid,EQRepresentation
from torchDataset import get_interaction_stream, get_interaction_stream_balanced
from tqdm import tqdm
import random

from SupervisedTrainer import SupervisedTrainer
from DockingTrainer import DockingTrainer

from DatasetGeneration import Protein, Complex

from torch.utils.tensorboard import SummaryWriter

def test(stream, trainer, epoch=0, theshold=0.5):
	TP, FP, TN, FN = 0, 0, 0, 0
	for data in tqdm(stream):
		tp, fp, tn, fn = trainer.eval_coef(data, theshold)
		TP += tp
		FP += fp
		TN += tn
		FN += fn
	
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
	print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1} MCC: {MCC}')
	return Accuracy, Precision, Recall, MCC

def get_threshold(stream, trainer):
	all_pred = []
	all_target = []
	for data in tqdm(stream):
		pred, target = trainer.eval(data)
		all_pred.append(pred.clone())
		all_target.append(target)

	all_pred = torch.cat(all_pred, dim=0)
	all_target = torch.cat(all_target, dim=0)
	sorted_pred, perm = torch.sort(all_pred)
	sorted_target = all_target[perm]
	target_true = (sorted_target == 1.0).to(dtype=torch.float32)
	target_false = (sorted_target == 0.0).to(dtype=torch.float32)
	cum_true = torch.cumsum(target_true, dim=0)
	cum_false = torch.cumsum(target_false.flip(dims=(0,)), dim=0).flip(dims=(0,))
	cum = cum_true + cum_false
	m, idx = torch.max(cum, dim=0)
	
	return sorted_pred[idx].item()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='Debug', type=str)
	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')

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

	train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=25, shuffle=True)
	valid_stream = get_interaction_stream('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=25)
	
	logger = SummaryWriter(Path('Log')/Path(args.experiment))

	if args.model() == 'resnet':
		model = CNNInteractionModel().cuda()
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = SupervisedTrainer(model, optimizer, type='int')

	elif args.model() == 'docker':
		repr = EQRepresentation(bias=False)
		# repr = EQRepresentationSid()
		scoring_model = EQScoringModel(repr=repr).to(device='cuda')
		if not(args.pretrain is None):
			trainer = DockingTrainer(scoring_model, None, type='pos')
			trainer.load_checkpoint(Path('Log')/Path(args.pretrain)/Path('dock_ebm.th'))
		
		model = EQInteractionF(scoring_model).cuda()
		optimizer = optim.Adam([{'params': scoring_model.parameters(), 'lr':0.0},
								{'params': model.F0, 'lr':10.0}])
		trainer = DockingTrainer(model, optimizer, type='int', omega=1.0)
	
	if args.cmd() == 'train':
		iter = 0
		for epoch in range(args.num_epochs):
			losses = []
			for data in tqdm(train_stream):
				log_dict = trainer.step(data)
				logger.add_scalar("DockFI/Train/Loss", log_dict["Loss"], iter)
				logger.add_scalars("DockFI/Train/F0", {"F0": log_dict["F0"], "<P>": torch.mean(log_dict["P"])}, iter)
				logger.add_histogram("DockFI/Train/P", log_dict["P"], iter)
				logger.add_scalar("DockFI/Train/Pstd", torch.std(log_dict["P"]).item(), iter)
				logger.add_scalars("DockFI/Train/Losses", {"BCE": log_dict["LossBCE"], "Reg": log_dict["LossReg"]}, iter)
				for i, param in enumerate(trainer.model.repr.parameters()):
					if param.ndimension() > 0:
						logger.add_histogram(f'DockFI/Model/{i}', param.detach().cpu(), iter)

				iter += 1
				losses.append(log_dict["Loss"])
			
			if torch.abs(torch.mean(log_dict["P"]) - log_dict["F0"])<torch.std(log_dict["P"]):
				optimizer.param_groups[0]['lr'] = 0.0
				optimizer.param_groups[1]['lr'] = 1e-2
				trainer.omega = 1e-3
			# else:
			# 	trainer.omega = 1.0
			# if iter>50:
			# 	optimizer.param_groups[0]['lr'] = 1e-3
			# 	optimizer.param_groups[1]['lr'] = 1e-3
			
			print(f'Loss {np.mean(losses)}')
			print(model.F0.item())
			Accuracy, Precision, Recall, MCC = test(valid_stream, trainer, epoch=epoch, theshold=0.5)


			logger.add_scalar("DockFI/Valid/acc", Accuracy, epoch)
			logger.add_scalar("DockFI/Valid/prec", Precision, epoch)
			logger.add_scalar("DockFI/Valid/rec", Recall, epoch)
			logger.add_scalar("DockFI/Valid/MCC", MCC, epoch)
			
			torch.save(model.state_dict(), Path('Log')/Path(args.experiment)/Path('model.th'))
		
	
	elif args.cmd() == 'test':
		trainer.load_checkpoint(logger.log_dir / Path('model.th'))
		test_stream = get_interaction_stream('DatasetGeneration/interaction_data_test.pkl', batch_size=32, max_size=1000)
		print('Validation:')
		Accuracy, Precision, Recall = test(valid_stream, trainer, 0)
		print('Test:')
		Accuracy, Precision, Recall = test(test_stream, trainer, 0)
