import torch
from torch import optim
import argparse
from pathlib import Path

import numpy as np

from Models import CNNInteractionModel, EQInteraction, EQRepresentation, EQScoringModel
from torchDataset import get_interaction_stream, get_interaction_stream_balanced
from tqdm import tqdm
import random

from SupervisedTrainer import SupervisedTrainer

from DatasetGeneration import Protein, Complex
from Logger import Logger

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='DebugCNNInter', type=str)
	args = parser.parse_args()

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=32, max_size=100, shuffle=True)
	valid_stream = get_interaction_stream('DatasetGeneration/interaction_data_valid.pkl', batch_size=32, max_size=100)

	model = CNNInteractionModel().cuda()
	# repr = EQRepresentation().cuda()
	# scoring = EQScoringModel().cuda()
	# model = EQInteraction(repr, scoring).cuda()
	optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
	trainer = SupervisedTrainer(model, optimizer)

	logger = Logger.new(Path('Log')/Path(args.experiment))
		
	for epoch in range(100):
		for data in tqdm(train_stream):
			loss = trainer.step(data)
			logger.log_train(loss)
		
		TP, FP, TN, FN = 0, 0, 0, 0
		for data in tqdm(valid_stream):
			tp, fp, tn, fn = trainer.eval(data)
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
		logger.log_valid_inter(Accuracy, Precision, Recall)

		if (epoch+1)%10 == 0:
			torch.save(model.state_dict(), logger.log_dir / Path('model.th'))
