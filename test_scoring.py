import torch
from torch import optim

import numpy as np

from Models import EQScoringModel, EQDockModel
from torchDataset import get_dataset_stream
from DatasetGeneration import rotate_ligand

from tqdm import tqdm

def run_docking_model(data, docker):
	receptor, ligand, translation, rotation = data
	
	receptor = receptor.to(device='cuda', dtype=torch.float)
	ligand = ligand.to(device='cuda', dtype=torch.float)
	docker.eval()
	angle, x, y, score = docker(receptor, ligand)


	# box_size = int(ligand.size(1))
	# field_size = box_size*3
	# field = np.zeros( (field_size, field_size) )
	
	# rligand = ligand.numpy()
	# this_field = field[:, i*field_size: (i+1)*field_size]
	# this_field[ int(field_size/2 - box_size/2): int(field_size/2 + box_size/2),
	# 			int(field_size/2 - box_size/2): int(field_size/2 + box_size/2)] += all_receptors[i,:,:].numpy()
	
	# this_field[ int(field_size/2 - box_size/2 + dx): int(field_size/2 + dx + box_size/2),
	# 			int(field_size/2 - box_size/2 + dy): int(field_size/2 + dy + box_size/2) ] += 2*rligand
	
	# receptors[:,i*box_size:(i+1)*box_size] = all_receptors[i,:,:]
	# ligands[:,i*box_size:(i+1)*box_size] = all_ligands[i,:,:]
	
	# f = plt.figure(figsize=(12,6))
	# plt.subplot(3,1,1)
	# plt.imshow(field)
	# plt.subplot(3,1,2)
	# plt.imshow(receptors)
	# plt.subplot(3,1,3)
	# plt.imshow(ligands)
	# plt.tight_layout()
	# plt.show()
	
	return

if __name__=='__main__':
	
	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(1)

	train_stream = get_dataset_stream('DatasetGeneration/toy_dataset_1000.pkl', batch_size=1)
	valid_stream = get_dataset_stream('DatasetGeneration/toy_dataset.pkl', batch_size=1)
	
	model = EQScoringModel().to(device='cuda')
	model.load_state_dict(torch.load('Log/dock_scoring_eq.th'))
	docker = EQDockModel(model)

	for data in tqdm(valid_stream):
		run_docking_model(data, docker)
		with open('Log/log_test_scoring_eq.txt', 'w') as fout:
			fout.write('Epoch\tLoss\n')
	

	
		

	

			





