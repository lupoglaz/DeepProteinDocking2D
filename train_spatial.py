import torch
from torch import optim

import numpy as np

from Models import EQSpatialDockModel
from torchDataset import get_dataset_stream
from DatasetGeneration import rotate_ligand

from tqdm import tqdm


def generate_ligands(ligands, rotations):
	rot_ligands = []
	for i in range(ligands.size(0)):
		rot_ligand = rotate_ligand(ligands[i,:,:].numpy(), rotations[i].item())
		rot_ligands.append(torch.from_numpy(rot_ligand))
		
	return torch.stack(rot_ligands, dim=0)

def run_spatial_model(data, model, train=True):
	receptor, ligand, translation, rotation = data

	#replicating receptors
	all_receptors = torch.cat([receptor, receptor], dim=0)

	#correct ligand rotations
	correct_ligands = generate_ligands(ligand, rotation)

	#randomly rotating ligands
	incorrect_rotations = torch.rand(ligand.size(0))*360.0
	incorrect_ligands = generate_ligands(ligand, incorrect_rotations)
	
	all_ligands = torch.cat([correct_ligands, incorrect_ligands], dim=0)

	#Translations for rotated ligands = 0
	incorrect_translations = torch.zeros(translation.size(0), 2, dtype=torch.double)
	all_translations = torch.cat([translation, incorrect_translations], dim=0)
	
	all_receptors = all_receptors.to(device='cuda', dtype=torch.float)
	all_ligands = all_ligands.to(device='cuda', dtype=torch.float)
	all_translations = all_translations.to(device='cuda', dtype=torch.float)
		
	if train:
		model.train()
		model.zero_grad()
	else:
		model.eval()

	t, hidden = model(all_receptors, all_ligands)
	
	half = int(t.size(0)/2)
	diff = torch.sqrt(torch.sum((t - all_translations)*(t - all_translations), dim=1))
	loss_translation = torch.mean(diff[:half])
	loss_translation_rot = torch.mean(diff[half:])
	loss = loss_translation/50.0 + loss_translation_rot/100.0
	
	if train:
		loss.backward()
		optimizer.step()
	else:
		print(t)
	
	return loss.item(), loss_translation.item(), loss_translation_rot.item()

if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/toy_dataset_1000.pkl', batch_size=10)
	valid_stream = get_dataset_stream('DatasetGeneration/toy_dataset.pkl', batch_size=10)
	
	model = EQSpatialDockModel().to(device='cuda')
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	with open('Log/log_train_spatial_eq.txt', 'w') as fout:
		fout.write('Epoch\tLoss\tTrans\tRot\n')
	with open('Log/log_valid_spatial_eq.txt', 'w') as fout:
		fout.write('Epoch\tLoss\tTrans\tRot\n')
	
	valid = []
	train = []
	num_angles = int(32)
	for epoch in range(100):
		loss = []
		for data in tqdm(train_stream):
			loss.append([run_spatial_model(data, model, train=True)])
		
		av_loss = np.average(loss, axis=0)[0,:]
		print('Epoch', epoch, 'Train Loss:', av_loss)
		with open('Log/log_train_spatial_eq.txt', 'a') as fout:
			fout.write('%d\t%f\t%f\t%f\n'%(epoch,av_loss[0],av_loss[1],av_loss[2]))

		loss = []
		for data in tqdm(valid_stream):
			loss.append([run_spatial_model(data, model, train=False)])
		
		av_loss = np.average(loss, axis=0)[0,:]
		print('Epoch', epoch, 'Valid Loss:', av_loss)
		with open('Log/log_valid_spatial_eq.txt', 'a') as fout:
			fout.write('%d\t%f\t%f\t%f\n'%(epoch,av_loss[0],av_loss[1],av_loss[2]))

	torch.save(model.state_dict(), 'Log/dock_spatial_eq.th') 
		

	

			





