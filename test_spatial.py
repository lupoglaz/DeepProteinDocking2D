import torch
from torch import optim

import numpy as np

from Models import EQSpatialDockModel
from torchDataset import get_dataset_stream
from DatasetGeneration import rotate_ligand

from tqdm import tqdm
import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

def generate_ligands(ligands, rotations):
	rot_ligands = []
	for i in range(ligands.size(0)):
		rot_ligand = rotate_ligand(ligands[i,:,:].numpy(), rotations[i].item())
		rot_ligands.append(torch.from_numpy(rot_ligand))
		
	return torch.stack(rot_ligands, dim=0)

def run_spatial_model(data, model, train=True):
	model.eval()
	receptor, ligand, translation, rotation = data
	print(rotation)	
	print(translation)
	#replicating receptors
	all_receptors = torch.cat([receptor, receptor], dim=0)

	#correct ligand rotations
	correct_ligands = generate_ligands(ligand, rotation)

	#randomly rotating ligands
	incorrect_rotations = torch.rand(ligand.size(0))*360.0
	incorrect_ligands = generate_ligands(ligand, incorrect_rotations)
	
	all_rotations = torch.cat([rotation, incorrect_rotations], dim=0)
	all_ligands = torch.cat([correct_ligands, incorrect_ligands], dim=0)

	#Translations for rotated ligands = 0
	incorrect_translations = torch.zeros(translation.size(0), 2, dtype=torch.double)
	all_translations = torch.cat([translation, incorrect_translations], dim=0)

	all_receptors = all_receptors.to(device='cpu', dtype=torch.float)
	all_ligands = all_ligands.to(device='cpu', dtype=torch.float)
	all_translations = all_translations.to(device='cpu', dtype=torch.float)
	all_rotations = all_rotations.to(device='cpu', dtype=torch.float)

	t, hidden = model(all_receptors, all_ligands)

	all_receptors = torch.cat([all_receptors, all_receptors], dim=0)
	all_ligands = torch.cat([all_ligands, all_ligands], dim=0)
	all_rotations = torch.cat([all_rotations, all_rotations], dim=0)
	all_translations = torch.cat([all_translations, t], dim=0)
	print(all_translations)

	batch_size = int(all_receptors.size(0))
	box_size = int(all_ligands.size(1))
	field_size = box_size*3
	field = np.zeros( (field_size, batch_size*field_size) )
	receptors = np.zeros( (box_size, batch_size*box_size) )
	ligands = np.zeros( (box_size, batch_size*box_size) )
	
	for i in range(batch_size):
		rligand = all_ligands[i,:,:].numpy()
		dx, dy = int(all_translations[i,0].item()), int(all_translations[i,1].item())
		this_field = field[:, i*field_size: (i+1)*field_size]
		this_field[ int(field_size/2 - box_size/2): int(field_size/2 + box_size/2),
					int(field_size/2 - box_size/2): int(field_size/2 + box_size/2)] += all_receptors[i,:,:].numpy()
		
		this_field[ int(field_size/2 - box_size/2 + dx): int(field_size/2 + dx + box_size/2),
					int(field_size/2 - box_size/2 + dy): int(field_size/2 + dy + box_size/2) ] += 2*rligand
		
		receptors[:,i*box_size:(i+1)*box_size] = all_receptors[i,:,:]
		ligands[:,i*box_size:(i+1)*box_size] = all_ligands[i,:,:]
	
	f = plt.figure(figsize=(12,6))
	plt.subplot(3,1,1)
	plt.imshow(field)
	plt.subplot(3,1,2)
	plt.imshow(receptors)
	plt.subplot(3,1,3)
	plt.imshow(ligands)
	plt.tight_layout()
	plt.show()
	
	return

if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/toy_dataset_1000.pkl', batch_size=1)
	valid_stream = get_dataset_stream('DatasetGeneration/toy_dataset.pkl', batch_size=1)
	
	model = EQSpatialDockModel().to(device='cuda')
	model.load_state_dict(torch.load('Log/dock_spatial_eq.th'))
	model = model.to(device='cpu')

	for data in tqdm(train_stream):
		run_spatial_model(data, model, train=True)
	
	for data in tqdm(valid_stream):
		run_spatial_model(data, model, train=False)
	
	
	
		

	

			





