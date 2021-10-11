import torch
from torch import nn
from torch.autograd import Function
import numpy as np

from .Convolution import ProteinConv2D
from .EQRepresentation import EQRepresentation, init_weights
from e2cnn import gspaces
from e2cnn import nn as e2nn
from math import *

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

class RMSDIndicator(nn.Module):
	def __init__(self):
		super(RMSDIndicator, self).__init__()

	def get_XC(self, protein):
		"""
		Analog of inertia tensor and center of mass for rmsd calc. Return 2/W * X and C
		"""
		batch_size = protein.size(0)
		size = protein.size(2)
		X = torch.zeros(batch_size, 2, 2, device=protein.device)
		C = torch.zeros(batch_size, 2, device=protein.device)
		x_i = (torch.arange(size).unsqueeze(dim=0).to(device=protein.device) - size/2.0).repeat(size, 1)
		y_i = (torch.arange(size).unsqueeze(dim=1).to(device=protein.device) - size/2.0).repeat(1, size)
		x_i = x_i.unsqueeze(dim=0).repeat(batch_size, 1, 1)
		y_i = y_i.unsqueeze(dim=0).repeat(batch_size, 1, 1)
		
		mask = (protein > 0.5).to(dtype=torch.float32).squeeze()
		W = torch.sum(mask) + 1E-5
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

	def forward(self, ligand, translation, rotation, coords):
		with torch.no_grad():
			X, C = self.get_XC(ligand)
		
		X = X.unsqueeze(dim=1)
		C = C.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
		
		#Translations
		T = coords[:,1:,:,:,:] - translation.unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
		T = T.transpose(1,2).transpose(2,3).transpose(3,4)

		#Rotations
		T0 = torch.stack([torch.cos(rotation), -torch.sin(rotation)], dim=1)
		T1 = torch.stack([torch.sin(rotation), torch.cos(rotation)], dim=1)
		R1 = torch.stack([T0, T1], dim=1).unsqueeze(dim=1)
		
		rotations = coords[0,0,:,0,0]
		T0 = torch.stack([torch.cos(rotations), -torch.sin(rotations)], dim=1)
		T1 = torch.stack([torch.sin(rotations), torch.cos(rotations)], dim=1)
		R2 = torch.stack([T0, T1], dim=1).unsqueeze(dim=0)
		
		R = R1.transpose(2,3) @ R2
		
		I = torch.diag(torch.ones(2, device=ligand.device)).unsqueeze(dim=0).unsqueeze(dim=1)
		#RMSD
		rmsd = torch.sum(T*T, dim=-1)
		rmsd = rmsd + torch.sum((I-R)*X, dim=(2,3)).unsqueeze(dim=2).unsqueeze(dim=3)
		rmsd = rmsd + 2.0*torch.sum(torch.sum(T.unsqueeze(dim=-1) * ((R2-R1).unsqueeze(dim=2).unsqueeze(dim=3)), dim=4) * C, dim=-1)
		return rmsd

class EQInteraction(nn.Module):
	def __init__(self, model):
		super(EQInteraction, self).__init__()
		self.repr = model.repr
		self.scorer = model.scorer
		self.conv = ProteinConv2D()
		self.indicator = RMSDIndicator()
		self.coords = None
		self.sigmoid = nn.Sigmoid()
		self.G0 = nn.Parameter(torch.tensor([-5.0], dtype=torch.float32))
		self.E0 = nn.Parameter(torch.zeros(1, dtype=torch.float32))

	def forward(self, scores, angles, ligand=None):
		assert scores.ndimension()==4
		batch_size = scores.size(0)
		num_angles = scores.size(1)
		L = scores.size(2)

		if self.coords is None:
			t = torch.from_numpy(np.linspace(-L/2, L/2, num=L)).to(device='cuda', dtype=torch.float32)
			x = t.unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
			x = x.repeat(1, num_angles, L, 1)
			y = t.unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=3)
			y = y.repeat(1, num_angles, 1, L)
			a = angles.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
			a = a.repeat(1, 1, L, L)
			self.coords = torch.cat([a,x,y], dim=0).unsqueeze(dim=0)
		
		with torch.no_grad():
			norm, _ = torch.min(scores.view(batch_size, num_angles*L*L), dim=1)
			norm = norm.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
		
		P = torch.exp(-scores+norm)
		coords = torch.sum(self.coords * P.unsqueeze(dim=1), dim=(2,3,4))/(torch.sum(P, dim=(1,2,3)).unsqueeze(dim=1))
				
		rmsd = self.indicator(ligand, coords[:,1:], coords[:,0], self.coords)
		indicator = torch.exp( -rmsd/9.0 )
		
		interaction = torch.sum( indicator * P, dim=(1,2,3))/torch.sum( P, dim=(1,2,3))
		return interaction

class SharpLoss(nn.Module):
	def __init__(self, trivial_penalty=0.01):
		super(SharpLoss, self).__init__()
		self.relu = nn.ReLU()
		self.trivial_penalty = trivial_penalty
		self.sigma = 1E-5
	
	def forward(self, pred, target):
		pred = pred.squeeze()
		target = target.squeeze()
		# assert pred.ndimension() == 1
		# assert target.ndimension() == 1
		
		label = 2*(target - 0.5)
		loss_pos = self.relu(pred) * (-self.relu(label))
		loss_neg = self.relu(-pred) * (-self.relu(-label))
		# print(pred)
		print('Label:', label.squeeze().cpu())
		# print(loss_pos)
		# print(loss_neg)
		# sys.exit()
		loss_trivial = 1.0/(pred*pred + 1.0)*self.trivial_penalty
		loss = loss_pos + loss_neg #+ loss_trivial #+ self.sigma*pred*pred
		return loss.mean()

class RankingLoss(nn.Module):
	def __init__(self):
		super(RankingLoss, self).__init__()
		self.relu = nn.ReLU()
		self.delta = 0.0

	def forward(self, pred, target):
		pred = pred.squeeze()
		target = target.squeeze()
		# assert pred.ndimension() == 1
		# assert target.ndimension() == 1
		# print(pred, target)
		batch_size = pred.size(0)
		loss = None
		N = 0
		for i in range(batch_size):
			for j in range(batch_size):
				# print(target[i].item(), target[j].item(), (pred[j] - pred[i]).item())
				if target[i]==0 and target[j]==1:
					if loss is None:
						loss = self.relu(self.delta - (pred[i] - pred[j]))
					else:
						loss += self.relu(self.delta - (pred[i] - pred[j]))
					N = N + 1

				elif target[i]==1 and target[j]==0:
					if loss is None:
						loss = self.relu(self.delta - (pred[j] - pred[i]))
					else:
						loss += self.relu(self.delta - (pred[j] - pred[i]))
					N = N + 1
				
		loss_trivial = (1.0/(pred*pred + 1e-5)).mean()

		if N>0:
			return loss_trivial + loss/float(N) 
		else:
			return None



class EQInteractionF(nn.Module):
	def __init__(self, model):
		super(EQInteractionF, self).__init__()
		self.repr = model.repr
		self.scorer = model.scorer
		self.conv = ProteinConv2D()
		self.sigmoid = nn.Sigmoid()
		self.batchnorm = nn.BatchNorm1d(1)
		self.F0 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
		
	def forward(self, scores, angles, ligand=None):
		assert scores.ndimension()==4
		batch_size = scores.size(0)
		num_angles = scores.size(1)
		L = scores.size(2)
				
		
		scores_flat = scores.view(batch_size, num_angles*L*L)
		P = -torch.logsumexp(-scores_flat, dim=-1)
		deltaP = P - self.F0
		return self.sigmoid(-deltaP), deltaP

class SidInteraction(nn.Module):
	def __init__(self, model):
		super(SidInteraction, self).__init__()
		self.repr = model.repr
		self.scorer = model.scorer
		self.conv = ProteinConv2D()
		self.coords = None
		self.softmax = torch.nn.Softmax2d()
		self.kernel = 5
		self.pad = self.kernel//2
		self.stride = 1
		self.dilation = 1
		self.conv3D = nn.Sequential(
			nn.Conv3d(1, 4, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=False),
			nn.ReLU(),
			nn.Conv3d(4, 1, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=False),
			nn.ReLU(),
		)

	def forward(self, scores, angles):
		assert scores.ndimension()==4
		batch_size = scores.size(0)
		num_angles = scores.size(1)
		L = scores.size(2)

		E = -scores
		P = self.softmax(-E)
		Esig = torch.sigmoid(-E)
		B = self.conv3D(Esig.unsqueeze(1)).squeeze()
		B = B.view(batch_size, num_angles, L, L)
		pred_interact = torch.sum(P * B, dim=(1,2,3)) / (torch.sum(P * B, dim=(1,2,3)) + Z/C)
		
		return pred_interact