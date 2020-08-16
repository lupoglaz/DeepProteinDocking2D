import torch
from torch import nn
from torch.autograd import Function
import numpy as np

from .multiplication import ImageCrossMultiply
from .multiplication_v2 import ImageCrossMultiplyV2
from .convolution import ProteinConv2D
from .eq_spatial_model import EQRepresentation
from e2cnn import gspaces
from e2cnn import nn as e2nn
from math import *

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

class EQScoringModel(nn.Module):
	def __init__(self, num_features=1, prot_field_size=50):
		super(EQScoringModel, self).__init__()
		self.prot_field_size = prot_field_size
				
		self.mult = ImageCrossMultiply()
		self.repr = EQRepresentation()		

		self.scorer = nn.Sequential(
			nn.ReLU(),
			nn.Linear(36,16),
			nn.ReLU(),
			nn.Linear(16,1)
		)

	def forward(self, receptor, ligand, T):
		rec_feat = self.repr(receptor.unsqueeze(dim=1)).tensor
		lig_feat = self.repr(ligand.unsqueeze(dim=1)).tensor
			
		pos_repr = self.mult(rec_feat, lig_feat, T)
				
		score = self.scorer(pos_repr)
		# norm = torch.sqrt((pos_repr*pos_repr).sum(dim=-1) + 1E-5)
		# pos_repr = pos_repr/norm
		return pos_repr

class EQScoringModelV2(nn.Module):
	def __init__(self, num_features=1, prot_field_size=50):
		super(EQScoringModelV2, self).__init__()
		self.prot_field_size = prot_field_size
				
		self.mult = ImageCrossMultiplyV2()
		self.repr = EQRepresentation()		

		self.scorer = nn.Sequential(
			nn.ReLU(),
			nn.Linear(36,16),
			nn.ReLU(),
			nn.Linear(16,1)
		)

	def forward(self, receptor, ligand, alpha, dr):
		rec_feat = self.repr(receptor.unsqueeze(dim=1)).tensor
		lig_feat = self.repr(ligand.unsqueeze(dim=1)).tensor
			
		pos_repr, _, A = self.mult(rec_feat, lig_feat, alpha, dr)
				
		score = self.scorer(pos_repr)
		# norm = torch.sqrt((pos_repr*pos_repr).sum(dim=-1) + 1E-5)
		# pos_repr = pos_repr/norm
		return pos_repr

class EQDockModel(nn.Module):
	def __init__(self, scoring_model, prot_field_size=50, num_angles=120, max_conf=100):
		super(EQDockModel, self).__init__()
		self.prot_field_size = prot_field_size
				
		self.conv = ProteinConv2D()
		self.scoring_model = scoring_model
		
		self.num_angles = num_angles
		self.max_conf = max_conf
		self.top_list = []

		self.angles = []
		self.rotations = []
		for i in range(num_angles):
			angle = float(i*np.pi)/float(num_angles)
			self.rotations.append(torch.tensor([[cos(angle), sin(angle), 0],
											[-sin(angle), cos(angle), 0]],
											dtype=torch.float, device='cuda'))
			self.angles.append(angle)
		
	def rotate(self, repr, i):
		with torch.no_grad():
			curr_grid = nn.functional.affine_grid(self.rotations[i].unsqueeze(dim=0), size=repr.size(), align_corners=True)
			return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

	def update_top(self, scores, angle):
		assert scores.ndimension() == 2
		L = scores.size(0)/2
		#Getting top scoring translations
		top = []
		for i in range(self.max_conf):
			maxval_y, ind_y = torch.max(scores, dim=1, keepdim=False)
			maxval_x, ind_x = torch.max(maxval_y, dim=0)
			x = ind_x.item()
			y = ind_y[x].item()
			self.top_list.append((angle, x-L, y-L, scores[x,y].item()))
			scores[x,y] = 0.0
		
		#Resorting the top conformations and cutting the max number
		self.top_list.sort(key = lambda t: -t[3])
		self.top_list = self.top_list[:self.max_conf]

	def score(self, translations):
		batch_size = translations.size(0)
		num_features = translations.size(1)
		L = translations.size(2)
		translations = translations.view(batch_size, num_features, L*L)
		translations = translations.transpose(1,2).contiguous().view(batch_size*L*L, num_features)
		scores = self.scoring_model.scorer(translations).squeeze()
		return scores.view(L, L)	

	def forward(self, receptor, ligand):
		assert receptor.size(0)==1
		assert ligand.size(0)==1
		with torch.no_grad():
			rec_feat = self.scoring_model.repr(receptor.unsqueeze(dim=1)).tensor
			lig_feat = self.scoring_model.repr(ligand.unsqueeze(dim=1)).tensor
			
			for i in range(self.num_angles):
				lig_rot = self.rotate(lig_feat, i)
				res = self.conv(rec_feat, lig_rot)
				scores = nn.Sigmoid()(self.score(res))
				# plt.subplot(1,3,1)
				# plt.imshow(receptor[0,:,:].cpu().numpy())
				# plt.subplot(1,3,2)
				# plt.imshow(lig_rot[0,0,:,:].cpu().numpy())
				# plt.subplot(1,3,3)
				# plt.imshow(scores.cpu().numpy())
				# plt.colorbar()
				# plt.show()
				self.update_top(scores, self.angles[i])
		
		return self.top_list[0]

