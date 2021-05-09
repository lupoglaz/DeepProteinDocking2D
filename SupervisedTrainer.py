import torch
from torch import optim
import torch.nn as nn
import numpy as np

from Models import ProteinConv2D
from tqdm import tqdm
import random
from math import cos, sin
import numpy as np

from matplotlib import pylab as plt

class SupervisedTrainer:
	def __init__(self, model, optimizer, device='cuda'):
		self.model = model
		self.optimizer = optimizer
		self.device = device
		self.loss = nn.BCELoss()

	def rotate(self, repr):
		with torch.no_grad():
			alpha = torch.rand(repr.size(0), dtype=torch.float32, device=repr.device)*2.0*np.pi
			T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
			T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
			R = torch.stack([T0, T1], dim=1)
			curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
			return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

	def step(self, receptor, ligand, interaction):
		receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		receptor = self.rotate(receptor)
		ligand = self.rotate(ligand)
		interaction = interaction.to(device=self.device).unsqueeze(dim=1)

		self.model.train()
		self.model.zero_grad()
		pred = self.model(receptor, ligand)
		loss = self.loss(pred, interaction)

		loss.backward()
		self.optimizer.step()
		return loss.item(),

	def eval(self, receptor, ligand, interaction, threshold=0.5):
		receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		self.model.eval()
		with torch.no_grad():
			pred = self.model(receptor, ligand)
			TP = 0
			FP = 0
			TN = 0
			FN = 0
			for i in range(pred.size(0)):
				p = pred[i].item()
				a = interaction[i].item()
				if p>=threshold and a>=threshold:
					TP += 1
				elif p>=threshold and a<threshold:
					FP += 1
				elif p<threshold and a>=threshold:
					FN += 1
				elif p<threshold and a<threshold:
					TN += 1
		
		return TP, FP, TN, FN