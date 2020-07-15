import torch
import torch.nn as nn
import os
import sys
import numpy as np
from torch.autograd import Function

import seaborn
from matplotlib import pylab as plt

from decomposition import HarmonicDecomposition, gaussian
import scipy
import math
from torch.nn.functional import affine_grid, grid_sample


class ProteinConvRotational(nn.Module):
	def __init__(self, box_size=50, N=20, alpha=2.0, lam=0.25):
		super(ProteinConvRotational, self).__init__()
			
	
	def forward(self, v1_coef, v2_coef):
		batch_size = v1_coef.size(0)
		num_features = v1_coef.size(1)
		N = v1_coef.size(2)
		L = v1_coef.size(3)
				
		cv1 = v1_coef.view(batch_size*num_features*N, L, 2)
		cv2 = v2_coef.view(batch_size*num_features*N, L, 2)
		re = cv1[:,:,0]*cv2[:,:,0] + cv1[:,:,1]*cv2[:,:,1]
		im = -cv1[:,:,0]*cv2[:,:,1] + cv1[:,:,1]*cv2[:,:,0]
		cconv = torch.stack([re, im], dim=2)
		circ_volume = torch.ifft(cconv, 1, normalized=True)

		circ_volume = circ_volume.view(batch_size, num_features, N, L, 2)
		circ_volume = circ_volume[:,:,:,:,0]*circ_volume[:,:,:,:,0] + circ_volume[:,:,:,:,1]*circ_volume[:,:,:,:,1]
		circ_volume = torch.sum(circ_volume, dim=2) #sum over radial k=0..N part
		return circ_volume
		

if __name__=='__main__':
	dec = HarmonicDecomposition(box_size=50, N=30, L=20, alpha=2.0, lam=0.50)

	a = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	b = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	R = 10
	gaussian(a, (R*math.cos(math.pi/4), R*math.sin(math.pi/4)), 2.0)
	gaussian(b, (R*math.cos(3*math.pi/2), R*math.sin(3*math.pi/2)), 2.0)
	
	a = a.unsqueeze(dim=0).unsqueeze(dim=1)
	b = b.unsqueeze(dim=0).unsqueeze(dim=1)

	pcr = ProteinConvRotational()
	a_coef = dec.euclid2basis(a)
	b_coef = dec.euclid2basis(b)
	c = pcr(b_coef, a_coef)

	

	batch_size = 20
	a = a.repeat(batch_size, 1, 1, 1)
	b = b.repeat(batch_size, 1, 1, 1)
	translation = torch.zeros(batch_size, 2)
	angle = torch.from_numpy(np.linspace(0, 2*np.pi, batch_size)).to(dtype=torch.float32)
	mat_col1 = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros(batch_size)], dim=1)
	mat_col2 = torch.stack([-torch.sin(angle), torch.cos(angle), torch.zeros(batch_size)], dim=1)
	mat_col3 = torch.cat([translation, torch.ones(batch_size, 1)], dim=1)
	mat = torch.stack([mat_col1, mat_col2, mat_col3], dim=2)
	grid = affine_grid(mat[:, :2, :3], a.size())
	b_rot = grid_sample(b, grid, mode='nearest')
	
	

	print(mat.size())
	print(mat[0,:,:])
	print(mat[10,:,:])
	print(mat[19,:,:])
	print(b_rot.size())

	f = plt.figure(figsize=(4,12))
	plt.subplot(3,1,1)
	plt.imshow(a[0,0,:,:].numpy())
	plt.colorbar()
	plt.subplot(3,1,2)
	plt.imshow(b[10,0,:,:].numpy())
	plt.colorbar()
	
	conv = (b_rot*a).sum(dim=3).sum(dim=2)
	plt.subplot(3,1,3)
	plt.plot(np.linspace(0, 360, c.size(2)), 2*math.sqrt(np.pi)*batch_size*c.squeeze().numpy(), label='harm conv')
	plt.plot(np.linspace(0, 360, conv.size(0)), conv.squeeze().numpy(), label='eucl conv')
	plt.legend()

	plt.tight_layout()
	plt.show()