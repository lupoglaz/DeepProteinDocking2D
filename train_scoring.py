import torch
from torch import optim

import numpy as np

from Models import EQScoringModel
from torchDataset import get_dataset_stream
from DatasetGeneration import rotate_ligand

from tqdm import tqdm


def generate_ligands(ligands, rotations):
	rot_ligands = []
	for i in range(ligands.size(0)):
		rot_ligand = rotate_ligand(ligands[i,:,:].numpy(), rotations[i].item())
		rot_ligands.append(torch.from_numpy(rot_ligand))
		
	return torch.stack(rot_ligands, dim=0)

def run_scoring_model(data, model, train=True):
	loss = torch.nn.CrossEntropyLoss()
	receptor, ligand, translation, rotation = data
	batch_size = receptor.size(0)*2

	#replicating receptors
	all_receptors = torch.cat([receptor, receptor], dim=0)

	#correct ligand rotations
	correct_ligands = generate_ligands(ligand, rotation)

	#randomly rotating ligands
	incorrect_rotations = torch.rand(ligand.size(0))*360.0
	incorrect_ligands = generate_ligands(ligand, incorrect_rotations)
	
	all_ligands = torch.cat([correct_ligands, incorrect_ligands], dim=0)

	#Translations for rotated ligands = 0
	incorrect_translations = torch.randint_like(translation, low=-receptor.size(2), high=receptor.size(2))
	all_translations = torch.cat([translation, incorrect_translations], dim=0)
	
	labels = torch.zeros(batch_size, dtype=torch.long, device='cuda')
	labels[:int(batch_size/2)] = 1

	all_receptors = all_receptors.to(device='cuda', dtype=torch.float)
	all_ligands = all_ligands.to(device='cuda', dtype=torch.float)
	all_translations = all_translations.to(device='cuda', dtype=torch.float)

	if train:
		model.train()
		model.zero_grad()
	else:
		model.eval()

	score = model(all_receptors, all_ligands, all_translations)
	L = loss(score, labels)
	
	if train:
		L.backward()
		optimizer.step()
	
	return L.item(),

if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/toy_dataset_1000.pkl', batch_size=10)
	valid_stream = get_dataset_stream('DatasetGeneration/toy_dataset.pkl', batch_size=10)
	
	model = EQScoringModel().to(device='cuda')
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	with open('Log/log_train_scoring_eq.txt', 'w') as fout:
		fout.write('Epoch\tLoss\n')
	with open('Log/log_valid_scoring_eq.txt', 'w') as fout:
		fout.write('Epoch\tLoss\n')
	
	valid = []
	train = []
	num_angles = int(32)
	for epoch in range(300):
		loss = []
		for data in tqdm(train_stream):
			loss.append([run_scoring_model(data, model, train=True)])
		
		av_loss = np.average(loss, axis=0)[0,:]
		print('Epoch', epoch, 'Train Loss:', av_loss)
		with open('Log/log_train_scoring_eq.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch,av_loss[0]))

		loss = []
		for data in tqdm(valid_stream):
			loss.append([run_scoring_model(data, model, train=False)])
		
		av_loss = np.average(loss, axis=0)[0,:]
		print('Epoch', epoch, 'Valid Loss:', av_loss)
		with open('Log/log_valid_scoring_eq.txt', 'a') as fout:
			fout.write('%d\t%f\n'%(epoch,av_loss[0]))

	torch.save(model.state_dict(), 'Log/dock_scoring_eq.th') 
		

	

			





