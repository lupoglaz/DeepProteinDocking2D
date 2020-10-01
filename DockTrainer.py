import torch
from torch import optim
import numpy as np

from Models import EQScoringModelV2, EQDockModel
from torchDataset import get_dataset_stream
from tqdm import tqdm
import random

from matplotlib import pylab as plt

from Models import ProteinConv2D

class DockTrainer:
	def __init__(self, model, optimizer, buffer, device='cuda', num_samples=10, weight=1.0, step_size=10.0, sample_steps=10):
		self.model = model
		self.optimizer = optimizer
		self.buffer = buffer

		self.weight = weight
		self.device = device

		self.plot_idx = 0
		self.conv = ProteinConv2D()

	def dock_spatial(self, rec_repr, lig_repr):
		
		translations = self.conv(rec_repr, lig_repr)
		
		batch_size = translations.size(0)
		num_features = translations.size(1)
		L = translations.size(2)

		translations = translations.view(batch_size, num_features, L*L)
		translations = translations.transpose(1,2).contiguous().view(batch_size*L*L, num_features)
		scores = self.model.scorer(translations).squeeze()
		scores = scores.view(batch_size, L, L)

		plt.imshow(scores[0,:,:].detach().cpu(), cmap='magma')
		plt.show()
		sys.exit()
		return None

	def dock_angular(self, rec_dec, lig_dec):
		pass

	def step_stoch(self, data, epoch=None):
		receptor, ligand, translation, rotation, pos_idx = data
		
		pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_alpha = rotation.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)*(np.pi/180.0)
		pos_dr = translation.to(device=self.device, dtype=torch.float32)

		batch_size = pos_rec.size(0)
		num_features = pos_rec.size(1)
		L = pos_rec.size(2)

		self.model.eval()
		rec_repr = self.model.repr(pos_rec).tensor
		lig_repr = self.model.repr(pos_lig).tensor
		neg_alpha, neg_dr = self.dock_spatial(rec_repr, lig_repr)

		self.requires_grad(True)
		self.model.train()
		self.model.zero_grad()
		
		pos_out = self.model(pos_rec, pos_lig, pos_alpha, pos_dr)
		L_p = (pos_out + self.weight * pos_out ** 2).mean()
		neg_out = self.model(neg_rec, neg_lig, neg_alpha, neg_dr)
		L_n = (-neg_out + self.weight * neg_out ** 2).mean()
		loss = L_p + L_n
		loss.backward()

		self.optimizer.step()
		self.buffer.push(pos_alpha, pos_dr, pos_idx)
		self.buffer.push(neg_alpha, neg_dr, neg_idx)
		
		return loss.item(),
	
