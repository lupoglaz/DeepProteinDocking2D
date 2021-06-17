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

class EQInteraction(nn.Module):
	def __init__(self, model):
		super(EQInteraction, self).__init__()
		self.repr = model.repr
		self.scorer = model.scorer
		self.conv = ProteinConv2D()
		self.coords = None
		self.sigmoid = nn.Sigmoid()
		self.G0 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
		self.E0 = nn.Parameter(torch.zeros(1, dtype=torch.float32))

	def forward(self, scores, angles):
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
		
		P = torch.sigmoid(self.E0 - scores).unsqueeze(dim=1)
		coords = torch.sum(self.coords * P, dim=(2,3,4))/torch.sum(P, dim=(2,3,4))
		coords = coords.unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
		
		#Change to RMSD
		indicator = torch.exp( -torch.sum((self.coords - coords)*(self.coords - coords), dim=1)/10.0 )
		
		#When computing this integral add 2 pi r dr
		dG = self.G0 - torch.log(1 + torch.sum(indicator * P.squeeze(), dim=(1,2,3)))
		interaction = self.sigmoid(dG)
		return interaction

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
		B = self.conv3D(E.unsqueeze(1)).squeeze()
		B = B.view(batch_size, num_angles, L, L)
		pred_interact = torch.sum(P * B, dim=(1,2,3)) / (torch.sum(P * B, dim=(1,2,3)) + 1)
		
		return pred_interact