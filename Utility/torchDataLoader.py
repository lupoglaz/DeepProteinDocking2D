import _pickle as pkl
import torch
from torch.utils.data import Dataset, RandomSampler
import random


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
		return receptor, ligand, rotation, translation

	def __len__(self):
		r"""
		Returns length of the dataset
		"""
		return self.dataset_size


class ToyInteractionDataset(Dataset):
	r"""
	"""
	def __init__(self, path, number_of_pairs=None):
		r"""
		"""
		self.path = path
		with open(self.path, 'rb') as fin:
			self.proteins, self.indices, self.labels = pkl.load(fin)

		self.data = []
		for i in range(len(self.labels)):
			receptor_index = self.indices[i][0]
			ligand_index = self.indices[i][1]
			receptor = self.proteins[receptor_index]
			ligand = self.proteins[ligand_index]
			label = self.labels[i]
			self.data.append([receptor, ligand, label])

		if not number_of_pairs:
			max_size = len(self.data)
		else:
			max_size = int(number_of_pairs + (number_of_pairs**2 - number_of_pairs)/2)

		random.shuffle(self.data)
		self.data = self.data[:max_size]
		self.dataset_size = len(list(self.data))

		print("Dataset file: ", self.path)
		print("Dataset size: ", self.dataset_size)

	def __getitem__(self, index):
		r"""
		"""
		receptor, ligand, interaction = self.data[index]
		return receptor, ligand, interaction

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


def get_interaction_stream(data_path, batch_size=1, shuffle=False, number_of_pairs=None, num_workers=0):
	dataset = ToyInteractionDataset(path=data_path, number_of_pairs=number_of_pairs)
	sampler = RandomSampler(dataset)
	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
	return trainloader


if __name__=='__main__':
	import timeit

	train_datapath = '../Datasets/interaction_train_100pool.pkl'
	valid_datapath = '../Datasets/interaction_valid_100pool.pkl'
	test_datapath = '../Datasets/interaction_test_100pool.pkl'

	start = timeit.default_timer()
	get_interaction_stream(train_datapath, max_size=1000)
	get_interaction_stream(valid_datapath, max_size=1000)
	get_interaction_stream(test_datapath, max_size=1000)
	end = timeit.default_timer()
	print('time to load datasets', end-start)
