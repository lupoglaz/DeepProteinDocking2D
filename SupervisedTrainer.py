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

class RMSDLoss(nn.Module):
	def __init__(self):
		super(RMSDLoss, self).__init__()

	def get_XC(self, protein):
		"""
		Analog of inertia tensor and center of mass for rmsd calc. Return 2/W * X and C
		"""
		batch_size = protein.size(0)
		size = protein.size(2)
		X = torch.zeros(batch_size, 2, 2)
		C = torch.zeros(batch_size, 2)
		x_i = (torch.arange(size).unsqueeze(dim=0) - size/2.0).repeat(size, 1)
		y_i = (torch.arange(size).unsqueeze(dim=1) - size/2.0).repeat(1, size)
		x_i = x_i.unsqueeze(dim=0).repeat(batch_size, 1, 1)
		y_i = y_i.unsqueeze(dim=0).repeat(batch_size, 1, 1)
		
		mask = (protein > 0.5).to(dtype=torch.float32)
		W = torch.sum(mask)
		x_i = x_i*mask
		y_i = y_i*mask
		#Inertia tensor
		X[:,0,0] = torch.sum(torch.sum(x_i*x_i, dim=-1), dim=-1)
		X[:,1,1] = torch.sum(torch.sum(y_i*y_i, dim=-1), dim=-1)
		X[:,0,1] = torch.sum(torch.sum(x_i*y_i, dim=-1), dim=-1)
		X[:,1,0] = torch.sum(torch.sum(y_i*x_i, dim=-1), dim=-1)
		#Center of mass
		C[:,0] = torch.sum(torch.sum(x_i, dim=-1), dim=-1)
		C[:,1] = torch.sum(torch.sum(x_i, dim=-1),dim=-1)
		return 2.0*X/W, C/W

	def forward(self, ligand, translation1, rotation1, translation2, rotation2):
		X, C = self.get_XC(ligand)
		T = T1 - T2
		
		R1 = torch.zeros(batch_size, 2, 2)
		R1[0,0] = torch.cos(rotation1)
		R1[1,1] = torch.cos(rotation1)
		R1[1,0] = torch.sin(rotation1)
		R1[0,1] = -torch.sin(rotation1)
		R2 = torch.zeros(2, 2)
		R2[0,0] = torch.cos(rotation2)
		R2[1,1] = torch.cos(rotation2)
		R2[1,0] = torch.sin(rotation2)
		R2[0,1] = -torch.sin(rotation2)
		R = R2.transpose(0,1) @ R1
		
		I = torch.diag(torch.ones(2))
		#RMSD
		rmsd = torch.sum(T*T)
		rmsd = rmsd + torch.sum((I-R)*X, dim=(0,1))
		rmsd = rmsd + 2.0*torch.sum(torch.sum(T.unsqueeze(dim=1) * (R1-R2), dim=0) * C, dim=0)
		return torch.sqrt(rmsd)


class SupervisedTrainer:
	def __init__(self, model, optimizer, device='cuda'):
		self.model = model
		self.optimizer = optimizer
		self.device = device
		if self.model.type=='int':
			self.loss = nn.BCELoss()
		elif self.model.type=='pos':
			self.loss = None

	def rotate(self, repr):
		with torch.no_grad():
			alpha = torch.rand(repr.size(0), dtype=torch.float32, device=repr.device)*2.0*np.pi
			T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
			T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
			R = torch.stack([T0, T1], dim=1)
			curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
			return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

	def step(self, receptor, ligand, target):
		receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		receptor = self.rotate(receptor)
		ligand = self.rotate(ligand)
		target = target.to(device=self.device).unsqueeze(dim=1)

		self.model.train()
		self.model.zero_grad()
		pred = self.model(receptor, ligand)
		loss = self.loss(pred, target)

		loss.backward()
		self.optimizer.step()
		return loss.item(),

	def eval(self, receptor, ligand, target, threshold=0.5):
		receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		self.model.eval()
		with torch.no_grad():
			pred = self.model(receptor, ligand)
			if self.model.type == 'int':
				TP = 0
				FP = 0
				TN = 0
				FN = 0
				for i in range(pred.size(0)):
					p = pred[i].item()
					a = target[i].item()
					if p>=threshold and a>=threshold:
						TP += 1
					elif p>=threshold and a<threshold:
						FP += 1
					elif p<threshold and a>=threshold:
						FN += 1
					elif p<threshold and a<threshold:
						TN += 1
				return TP, FP, TN, FN
			else:
				pass