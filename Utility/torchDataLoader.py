import os
import sys
# import _pickle as pkl
import _pickle as pkl

import torch
from torch.utils.data import Dataset, RandomSampler
import timeit


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
		receptor, ligand, rotation, translation = self.data[index]
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
		return receptor.unsqueeze(0), ligand.unsqueeze(0), interaction.unsqueeze(0)

	def __len__(self):
		r"""
		Returns length of the dataset
		"""
		return self.dataset_size


def get_docking_stream(data_path, batch_size=1, shuffle=False, max_size=None, num_workers=0):
	dataset = ToyDockingDataset(path=data_path, max_size=max_size)
	sampler = RandomSampler(dataset)
	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
	return trainloader


def get_interaction_stream(data_path, batch_size=1, shuffle=False, max_size=None, num_workers=0):
	start = timeit.default_timer()
	dataset = ToyInteractionDataset(path=data_path, max_size=max_size)
	end = timeit.default_timer()
	print('timer 1')
	print(end - start)

	start = timeit.default_timer()
	sampler = RandomSampler(dataset)
	end = timeit.default_timer()
	print('timer 2')
	print(end - start)

	start = timeit.default_timer()
	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
	end = timeit.default_timer()
	print('timer 3')
	print(end - start)
	return trainloader


if __name__=='__main__':

	datapath = '../Datasets/interaction_train_set400pool.pkl'

	get_interaction_stream(datapath, max_size=1000, num_workers=0)
	#

	# start = timeit.default_timer()
	# get_interaction_stream(datapath, max_size=1000, num_workers=0)
	# end = timeit.default_timer()
	# print('timer 1')
	# print(end - start)
	#
	# start = timeit.default_timer()
	# get_interaction_stream(datapath, max_size=1000, num_workers=4)
	# end = timeit.default_timer()
	# print('timer 2')
	# print(end - start)
