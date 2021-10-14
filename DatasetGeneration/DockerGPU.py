import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from random import uniform
from .Protein import Protein
from .Complex import Complex
from tqdm import tqdm

import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Models import ProteinConv2D

class DockerGPU:
	def __init__(self, num_angles=360, a00=1.0, a11=-1.0, a10=-1.0, boundary_size=1):
		# Features correspondance: 0-bulk, 1-boundary
		# 0 : 0 - 0, bulk-bulk
		# 1 : 0 - 1, bulk-boundary
		# 2 : 1 - 0, boundary-bulk
		# 3 : 1 - 1, boundary-boundary
		self.num_angles = num_angles
		self.a00 = a00
		self.a11 = a11
		self.a10 = a10
		self.boundary_size = boundary_size
		self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=num_angles)).to(device='cuda')
		self.conv = ProteinConv2D()

	def rotate(self, repr, angle):
		alpha = angle.detach()
		T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
		T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
		R = torch.stack([T0, T1], dim=1)
		curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=False)
		return nn.functional.grid_sample(repr, curr_grid, align_corners=False, mode='nearest')
	
	def dock_global(self, receptor, ligand):
		receptor.make_boundary(boundary_size=self.boundary_size)
		ligand.make_boundary(boundary_size=self.boundary_size)

		f_rec = torch.from_numpy(receptor.get_repr()).to(device='cuda')
		f_lig = torch.from_numpy(ligand.get_repr()).to(device='cuda')
		
		f_lig = f_lig.unsqueeze(dim=0).repeat(self.num_angles, 1, 1, 1)
		f_rec = f_rec.unsqueeze(dim=0).repeat(self.num_angles, 1, 1, 1)
		rot_lig = self.rotate(f_lig, self.angles)
		
		translations = self.conv(f_rec, rot_lig)
		return self.a00*translations[:,0,:,:] + self.a11*translations[:,3,:,:] + self.a10*(translations[:,1,:,:]+translations[:,2,:,:])
				

	def get_conformation(self, scores, receptor, ligand):
		minval_y, ind_y = torch.min(scores, dim=2, keepdim=False)
		minval_x, ind_x = torch.min(minval_y, dim=1)
		minval_angle, ind_angle = torch.min(minval_x, dim=0)
		x = ind_x[ind_angle].item()
		y = ind_y[ind_angle, x].item()
		
		best_score = scores[ind_angle, x, y].item()
		best_translation = [x-scores.size(1)/2.0, y-scores.size(1)/2.0]
		best_rotation = self.angles[ind_angle].item()
		
		res_cplx = Complex(receptor, ligand, best_rotation, best_translation)
		
		return best_score, res_cplx, [ind_angle, x, y]

	