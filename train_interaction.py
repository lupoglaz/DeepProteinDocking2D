import torch
from torch import optim

import numpy as np

from Models import CNNInteractionModel, EQInteraction, EQRepresentation, EQScoringModel
from torchDataset import get_interaction_stream, get_interaction_stream_balanced
from tqdm import tqdm
import random

from SupervisedTrainer import SupervisedTrainer

from DatasetGeneration import Protein, Complex

if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=32, max_size=100, shuffle=True)
	valid_stream = get_interaction_stream('DatasetGeneration/interaction_data_valid.pkl', batch_size=32, max_size=100)

	# model = CNNInteractionModel().cuda()
	repr = EQRepresentation().cuda()
	scoring = EQScoringModel().cuda()
	model = EQInteraction(repr, scoring).cuda()
	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
	trainer = SupervisedTrainer(model, optimizer)

	with open('Log/log_train_interaction.txt', 'w') as fout:
		fout.write('Epoch\tLoss\n')
	with open('Log/log_valid_interaction.txt', 'w') as fout:
		fout.write('Epoch\tAccuracy\tPrecision\tRecall\n')
	
	for epoch in range(100):
		loss = []
		for receptor, ligand, interaction in tqdm(train_stream):
			loss.append([trainer.step(receptor, ligand, interaction)])
			# break
		
		av_loss = np.average(loss, axis=0)[0,:]
		
		print('Epoch', epoch, 'Train Loss:', av_loss)
		with open('Log/log_train_interaction.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch,av_loss[0]))
		
		if (epoch+1)%10 == 0:
			torch.save(model.state_dict(), 'Log/inter.th')

		TP, FP, TN, FN = 0, 0, 0, 0
		for receptor, ligand, interaction in tqdm(valid_stream):
			tp, fp, tn, fn = trainer.eval(receptor, ligand, interaction)
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

		print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall}')
		with open('Log/log_valid_interaction.txt', 'a') as fout:
			fout.write('%d\t%f\t%f\t%f\n'%(epoch, Accuracy, Precision, Recall))
