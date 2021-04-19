
import os
import sys
import _pickle as pkl
import torch
from torch.utils.data import Dataset
import numpy as np
import random
random.seed(42)

def crop_collate(batch):
	r"""
	"""
	
	receptors = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
	ligands = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
	translations = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
	rotations = torch.cat(list(map(lambda x: x[3], batch)), dim=0)
	index = torch.tensor(list(map(lambda x: x[4], batch)), dtype=torch.long)
	return receptors, ligands, translations, rotations, index

class ToyDataset2D(Dataset):
	r"""
	"""
	def __init__(self, path='toy_dataset_1000.pkl', max_size=100):
		r"""
		"""
		self.path = path
		with open(self.path, 'rb') as fin:
			self.data = pkl.load(fin)

		self.data = self.data[:max_size]
		self.dataset_size = len(list(self.data))

		print ("Dataset file: ", self.path)
		print ("Dataset size: ", self.dataset_size)
		
	def __getitem__(self, index):
		r"""
		"""
		receptor, ligand, translation, rotation = self.data[index]
		return torch.from_numpy(receptor), torch.from_numpy(ligand), torch.tensor(translation), torch.tensor([rotation]), index

		
	def __len__(self):
		r"""
		Returns length of the dataset
		"""
		return self.dataset_size


def get_dataset_stream(data_path, batch_size = 10, shuffle = False, max_size=1000):
	dataset = ToyDataset2D(path=data_path, max_size=max_size)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle, collate_fn=crop_collate)
	return trainloader

if __name__=='__main__':
	from DatasetGeneration import rotate_ligand
	from DatasetGeneration import Protein, Complex
	import matplotlib.pylab as plt
	import seaborn as sea
	sea.set_style("whitegrid")

	stream = get_dataset_stream(data_path='DatasetGeneration/dataset_train.pkl')
	for data in stream:
		receptor, ligand, translation, rotation, index = data
		break
	print(index)
	batch_size = int(receptor.size(0))
	box_size = int(receptor.size(1))
	field_size = box_size*3
	field = np.zeros( (field_size, batch_size*field_size) )
	receptors = np.zeros( (box_size, batch_size*box_size) )
	ligands = np.zeros( (box_size, batch_size*box_size) )
	print(batch_size, box_size, field_size)
	for i in range(batch_size):
		rligand = rotate_ligand(ligand[i,:,:].numpy(), 180.0*rotation[i].item()/np.pi)
		dx, dy = int(translation[i,0].item()), int(translation[i,1].item())
		this_field = field[:, i*field_size: (i+1)*field_size]
		this_field[ int(field_size/2 - box_size/2): int(field_size/2 + box_size/2),
					int(field_size/2 - box_size/2): int(field_size/2 + box_size/2)] += receptor[i,:,:].numpy()
		
		this_field[ int(field_size/2 - box_size/2 + dx): int(field_size/2 + dx + box_size/2),
					int(field_size/2 - box_size/2 + dy): int(field_size/2 + dy + box_size/2) ] += 2*rligand
		
		receptors[:,i*box_size:(i+1)*box_size] = receptor[i,:,:]
		ligands[:,i*box_size:(i+1)*box_size] = ligand[i,:,:]
	
	f = plt.figure(figsize=(12,6))
	plt.subplot(3,1,1)
	plt.imshow(field)
	plt.subplot(3,1,2)
	plt.imshow(receptors)
	plt.subplot(3,1,3)
	plt.imshow(ligands)
	plt.tight_layout()
	plt.show()

	cplx = Complex(Protein(receptor[0,:,:].numpy()), Protein(ligand[0,:,:].numpy()), rotation[0].item(), translation[0,:].numpy())
	cplx.plot()
	plt.show()
