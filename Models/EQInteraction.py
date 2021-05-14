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

class GaussianHistogram(nn.Module):
	def __init__(self, bins, min, max, sigma):
		super(GaussianHistogram, self).__init__()
		self.bins = bins
		self.min = min
		self.max = max
		self.sigma = sigma
		self.delta = float(max - min) / float(bins)
		self.centers = (float(min) + self.delta * (torch.arange(bins).float() + 0.5)).unsqueeze(dim=0).to(device='cuda', dtype=torch.float32)

	def forward(self, x):
		# print(torch.min(x, dim=1))
		# print(torch.max(x, dim=1))
		# sys.exit()
		x = x.unsqueeze(dim=1) - self.centers.unsqueeze(dim=2)
		x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
		x = x.sum(dim=2)
		return x

class EQCompression(nn.Module):
	def __init__(self):
		super(EQCompression, self).__init__()
		r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
		self.feat_type_in = e2nn.FieldType(r2_act, 8*[r2_act.irreps['irrep_0']])
		self.feat_type_hid = e2nn.FieldType(r2_act, 16*[r2_act.irreps['irrep_0']]+4*[r2_act.irreps['irrep_1']]+2*[r2_act.irreps['irrep_2']])
		self.feat_type_out = e2nn.FieldType(r2_act, 8*[r2_act.irreps['irrep_0']])
				
		self.comp = nn.Sequential(
			
			# e2nn.GNormBatchNorm(self.feat_type_hid),
			e2nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			# e2nn.GNormBatchNorm(self.feat_type_hid),
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			## 50 -> 25
			e2nn.NormMaxPool(self.feat_type_hid, kernel_size=3, stride=2, padding=1),
			
			# e2nn.GNormBatchNorm(self.feat_type_hid),
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			# e2nn.GNormBatchNorm(self.feat_type_hid),
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			## 25 -> 13 (72 angles)
			e2nn.NormMaxPool(self.feat_type_hid, kernel_size=3, stride=2, padding=1),
			
			# e2nn.GNormBatchNorm(self.feat_type_hid),
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			# e2nn.GNormBatchNorm(self.feat_type_hid),
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			# ## 13 -> 7 (25 angles)
			# e2nn.NormMaxPool(self.feat_type_hid, kernel_size=3, stride=2, padding=1),
			
			# #e2nn.GNormBatchNorm(self.feat_type_hid),
			# e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			# e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			# #e2nn.GNormBatchNorm(self.feat_type_hid),
			# e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			# e2nn.NormNonLinearity(self.feat_type_hid, bias=False),
			
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size=3, padding=1, bias=False),
		)

		with torch.no_grad():
			self.comp.apply(init_weights)

	def forward(self, x):
		y = self.comp(x)
		return y


class EQInteraction(nn.Module):
	def __init__(self, representation, scoring, 
						num_angles=72, bins=256, min=-5, max=5, sigma=0.1):
		super(EQInteraction, self).__init__()
		self.repr_model = representation
		self.scoring_model = scoring
		self.comp_model = EQCompression()
		self.conv = ProteinConv2D()
		self.num_angles = num_angles
		self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=num_angles)).to(device='cuda', dtype=torch.float32)
		self.hist = GaussianHistogram(bins=bins, min=min, max=max, sigma=sigma)

		self.hist_model = nn.Sequential(
			nn.Linear(bins, int(bins/2)),
			nn.ReLU(),
			nn.Linear(int(bins/2), 1),
			nn.Sigmoid()
		)
		
	def rotate(self, repr, angle):
		alpha = angle.detach()
		T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
		T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
		R = torch.stack([T0, T1], dim=1)
		curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
		return nn.functional.grid_sample(repr, curr_grid, align_corners=True)
	
	def dock_global(self, rec_repr, lig_repr):
		batch_size = lig_repr.size(0)
		
		angles = self.angles.unsqueeze(dim=0).repeat(batch_size, 1)
		angles = angles.view(batch_size*self.num_angles)
		
		lig_repr = lig_repr.unsqueeze(dim=1).repeat(1, self.num_angles, 1, 1, 1)
		rec_repr = rec_repr.unsqueeze(dim=1).repeat(1, self.num_angles, 1, 1, 1)
		
		rec_repr = rec_repr.view(batch_size*self.num_angles, rec_repr.size(-3), rec_repr.size(-2), rec_repr.size(-1))
		lig_repr = lig_repr.view(batch_size*self.num_angles, lig_repr.size(-3), lig_repr.size(-2), lig_repr.size(-1))
				
		rot_lig = self.rotate(lig_repr, angles)
		translations = self.conv(rec_repr, rot_lig)
		translations = translations.view(batch_size, self.num_angles, translations.size(-3), translations.size(-2), translations.size(-1))
		return translations

	def score(self, translations):
		batch_size = translations.size(0)
		num_rot = translations.size(1)
		num_feat = translations.size(2)
		L = translations.size(3)

		translations = translations.transpose(1,2).contiguous()
		translations = translations.view(batch_size, num_feat, num_rot*L*L)
		translations = translations.transpose(1,2).contiguous().view(batch_size*num_rot*L*L, num_feat)
		scores = self.scoring_model.scorer(translations).squeeze()
		return scores.view(batch_size, num_rot, L, L)

	def histograms(self, scores):
		pass

	def forward(self, receptor, ligand):
		assert ligand.size(1)==receptor.size(1)
		assert ligand.ndimension()==4
		assert ligand.ndimension()==receptor.ndimension()
		batch_size = receptor.size(0)
		rec_repr = self.repr_model(receptor)
		lig_repr = self.repr_model(ligand)
		rec_comp = self.comp_model(rec_repr)
		lig_comp = self.comp_model(lig_repr)
		translations = self.dock_global(rec_comp.tensor, lig_comp.tensor)
		scores = self.score(translations)
		
		hist = self.hist(scores.view(batch_size, scores.size(1)*scores.size(2)*scores.size(3)))
		interaction = self.hist_model(hist)
					
		return interaction