import os
import sys
import _pickle as pkl
import torch
from torch.utils.data import Dataset, BatchSampler, WeightedRandomSampler, SequentialSampler
import numpy as np
# import random
# random.seed(42)

def crop_collate(batch):
	r"""
	"""
	receptors = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
	ligands = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
	if len(batch[0])>3:
		translations = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
		rotations = torch.cat(list(map(lambda x: x[3], batch)), dim=0)
		index = torch.tensor(list(map(lambda x: x[4], batch)), dtype=torch.long)
		return receptors, ligands, translations, rotations, index
	else:
		interactions = torch.cat(list(map(lambda x: x[2], batch)), dim=0)
		return receptors, ligands, interactions

class ToyDockingDataset(Dataset):
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

class ToyInteractionDataset(Dataset):
	r"""
	"""
	def __init__(self, path='toy_dataset_1000.pkl', max_size=100, printout=False):
		r"""
		"""
		self.path = path
		with open(self.path, 'rb') as fin:
			self.proteins, self.interactome = pkl.load(fin)

		if max_size<len(self.proteins):
			self.proteins = self.proteins[:max_size]
			self.interactome = self.interactome[:max_size, :max_size]

		self.interactome = torch.from_numpy(self.interactome).to(dtype=torch.float32)
		self.num_proteins = len(list(self.proteins))

		self.weights = []
		self.indexes = []
		weight = float(self.num_proteins*self.num_proteins)/float(torch.sum(self.interactome).item() + 1e-5)
		N_pos = 0
		N_neg = 0
		for i in range(self.num_proteins):
			for j in range(self.num_proteins):
				if i>j: continue
				self.indexes.append((i,j))
				if self.interactome[i,j] == 1:
					self.weights.append(1.0)
					N_pos += 1
				else:
					self.weights.append(1.0/float(weight))
					N_neg += 1

		self.dataset_size = len(self.indexes)
		if printout:
			print ("Dataset file: ", self.path)
			print ("Dataset size: ", self.dataset_size)
			print ("Number of proteins: ", len(self.proteins))
			print ("Positive weight: ", weight, "Number of pos/neg:", N_pos, N_neg)

	def __getitem__(self, index):
		r"""
		"""
		i, j =self.indexes[index]
		receptor = self.proteins[i]
		ligand = self.proteins[j]
		interaction = self.interactome[i,j]
		return torch.from_numpy(receptor), torch.from_numpy(ligand), torch.tensor([interaction])


	def __len__(self):
		r"""
		Returns length of the dataset
		"""
		return self.dataset_size


def get_docking_stream(data_path, batch_size = 10, shuffle = False, max_size=1000):
	dataset = ToyDockingDataset(path=data_path, max_size=max_size)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle, collate_fn=crop_collate)
	return trainloader

def get_interaction_stream_balanced(data_path, batch_size=10, max_size=1000):
	dataset = ToyInteractionDataset(path=data_path, max_size=max_size)
	sampler = BatchSampler(WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset.weights), replacement=True), batch_size, drop_last=False)
	trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=crop_collate, shuffle=False, pin_memory=True)
	return trainloader

def get_interaction_stream_balanced_singleordering(data_path, sampler=None, printout=False, batch_size=1, max_size=400):
	dataset = ToyInteractionDataset(path=data_path, max_size=max_size, printout=printout)
	# print( torch.FloatTensor(dataset.weights))
	# print(torch.stack((list(dataset.weights))))
	sampler = ResumableRandomSampler(dataset, replacement=True)
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=crop_collate, shuffle=False, pin_memory=True)
	torch.save(sampler.get_state(), "saved_samplerstate.pth")
	sampler.set_state(torch.load("saved_samplerstate.pth"))
	return loader, sampler

def get_interaction_stream(data_path, batch_size = 10, max_size=1000):
	dataset = ToyInteractionDataset(path=data_path, max_size=max_size)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=crop_collate, shuffle=False)
	return trainloader

if __name__=='__main__':
	# stream = get_docking_stream(data_path='DatasetGeneration/docking_data_train.pkl')
	# for data in stream:
	# 	receptor, ligand, translation, rotation, index = data
	# 	break
	# print(receptor.size())
	# print(ligand.size())
	# print(translation.size())
	# print(rotation.size())
	# print(index)
	#
	# stream = get_interaction_stream_balanced(data_path='DatasetGeneration/interaction_data_train.pkl', batch_size=64, max_size=100)
	# all_pos = 0
	# for data in stream:
	# 	receptor, ligand, interaction = data
	# 	all_pos += torch.sum(interaction).item()
	# print(all_pos)
	# print(receptor.size())
	# print(ligand.size())
	# print(interaction)

	stream, sampler = get_interaction_stream_balanced_singleordering(data_path='DatasetGeneration/interaction_data_train.pkl', batch_size=1, max_size=10)
	# print(stream)
	first_sampler = []
	for data in stream:
		receptor, ligand, interaction = data
		first_sampler.append(interaction)

	stream, _ = get_interaction_stream_balanced_singleordering(data_path='DatasetGeneration/interaction_data_train.pkl', sampler=sampler, batch_size=1, max_size=10)
	second_sampler = []
	for data in stream:
		receptor, ligand, interaction = data
		second_sampler.append(interaction)

	print(first_sampler)
	print(second_sampler)
