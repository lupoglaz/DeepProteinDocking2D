import os
import sys
import _pickle as pkl
import torch
from torch.utils.data import Dataset, RandomSampler
import numpy as np
# import random
# random.seed(42)


class ToyDockingDataset(Dataset):
	r"""
	"""
	def __init__(self, path, max_size=None):
		r"""
		"""
		self.path = path
		with open(self.path, 'rb') as fin:
			self.data = pkl.load(fin)

		if not max_size:
			max_size = len(self.data)
		self.data = self.data[:max_size]
		self.dataset_size = len(list(self.data))

		print ("Dataset file: ", self.path)
		print ("Dataset size: ", self.dataset_size)

	def __getitem__(self, index):
		r"""
		"""
		receptor, ligand, translation, rotation = self.data[index]
		return receptor.unsqueeze(0), ligand.unsqueeze(0), rotation.unsqueeze(0), translation.unsqueeze(0)


	def __len__(self):
		r"""
		Returns length of the dataset
		"""
		return self.dataset_size


class ToyInteractionDataset(Dataset):
	r"""
	"""
	def __init__(self, path, max_size=None):
		r"""
		"""
		self.path = path
		with open(self.path, 'rb') as fin:
			self.data = pkl.load(fin)

		if not max_size:
			max_size = len(self.data)
		self.data = self.data[:max_size]
		self.dataset_size = len(list(self.data))

		print ("Dataset file: ", self.path)
		print ("Dataset size: ", self.dataset_size)

	def __getitem__(self, index):
		r"""
		"""
		receptor, ligand, interaction = self.data[index]
		return receptor.unsqueeze(0), ligand.unsqueeze(0), torch.tensor(interaction).unsqueeze(0)

	def __len__(self):
		r"""
		Returns length of the dataset
		"""
		return self.dataset_size

# class ToyInteractionDataset(Dataset):
# 	r"""
# 	"""
# 	def __init__(self, path='toy_dataset_1000.pkl', max_size=100, printout=False):
# 		r"""
# 		"""
# 		self.path = path
# 		with open(self.path, 'rb') as fin:
# 			self.proteins, self.interactome = pkl.load(fin)
# 			print(self.proteins)
# 			print(self.interactome)
#
#
# 		if max_size<len(self.proteins):
# 			self.proteins = self.proteins[:max_size]
# 			self.interactome = self.interactome[:max_size, :max_size]
#
# 		self.interactome = torch.from_numpy(self.interactome).to(dtype=torch.float32)
# 		self.num_proteins = len(list(self.proteins))
#
# 		self.weights = []
# 		self.indexes = []
# 		weight = float(self.num_proteins*self.num_proteins)/float(torch.sum(self.interactome).item() + 1e-5)
# 		N_pos = 0
# 		N_neg = 0
# 		for i in range(self.num_proteins):
# 			for j in range(self.num_proteins):
# 				if i>j: continue
# 				self.indexes.append((i,j))
# 				if self.interactome[i,j] == 1:
# 					self.weights.append(1.0)
# 					N_pos += 1
# 				else:
# 					self.weights.append(1.0/float(weight))
# 					N_neg += 1
#
# 		self.dataset_size = len(self.indexes)
# 		if printout:
# 			print ("Dataset file: ", self.path)
# 			print ("Dataset size: ", self.dataset_size)
# 			print ("Number of proteins: ", len(self.proteins))
# 			print ("Positive weight: ", weight, "Number of pos/neg:", N_pos, N_neg)
#
# 	def __getitem__(self, index):
# 		r"""
# 		"""
# 		i, j =self.indexes[index]
# 		receptor = self.proteins[i]
# 		ligand = self.proteins[j]
# 		interaction = self.interactome[i,j]
# 		return torch.from_numpy(receptor), torch.from_numpy(ligand), torch.tensor([interaction])
#
#
# 	def __len__(self):
# 		r"""
# 		Returns length of the dataset
# 		"""
# 		return self.dataset_size


def get_docking_stream(data_path, batch_size=1, shuffle=False, max_size=None, num_workers=0):
	dataset = ToyDockingDataset(path=data_path, max_size=max_size)
	sampler = RandomSampler(dataset)
	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
	return trainloader


def get_interaction_stream(data_path, batch_size=1, shuffle=False, max_size=None, num_workers=0):
	dataset = ToyInteractionDataset(path=data_path, max_size=max_size)
	sampler = RandomSampler(dataset)
	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
	return trainloader


if __name__=='__main__':
	pass
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

	# stream, sampler = get_interaction_stream_balanced_singleordering(data_path='DatasetGeneration/interaction_data_train.pkl', batch_size=1, max_size=10)
	# # print(stream)
	# first_sampler = []
	# for data in stream:
	# 	receptor, ligand, interaction = data
	# 	first_sampler.append(interaction)
	#
	# stream, _ = get_interaction_stream_balanced_singleordering(data_path='DatasetGeneration/interaction_data_train.pkl', sampler=sampler, batch_size=1, max_size=10)
	# second_sampler = []
	# for data in stream:
	# 	receptor, ligand, interaction = data
	# 	second_sampler.append(interaction)
	#
	# print(first_sampler)
	# print(second_sampler)
