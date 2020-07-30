import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
class ImageMultiply(nn.Module):
	def __init__(self):
		super(ImageMultiply, self).__init__()

	def mabs(self, dx, dy):
		return np.abs(dx), np.abs(dy)
	
	def multiply(self, v1, v2, dx, dy, L):
		
		#all positive
		if dx>=0 and dy>=0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,dx:L, dy:L] * v2[:,0:L-dx, 0:L-dy]
		
		#one negative
		elif dx<0 and dy>=0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,0:L-dx, dy:L] * v2[:,dx:L, 0:L-dy]
		elif dx>=0 and dy<0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,dx:L, 0:L-dy] * v2[:,0:L-dx, dy:L]
		
		#all negative
		elif dx<0 and dy<0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,0:L-dx, 0:L-dy] * v2[:,dx:L, dy:L]

		return result.sum(dim=2).sum(dim=1).squeeze()

	def forward(self, receptor, ligand, T):
		batch_size = receptor.size(0)
		L = receptor.size(2)
		mults = []
		for i in range(batch_size):
			v1 = receptor[i,:,:,:].squeeze()
			v2 = ligand[i,:,:,:].squeeze()
			dx = int(T[i,0])
			dy = int(T[i,1])
			mults.append(self.multiply(v1,v2,dx,dy,L))
		return torch.stack(mults, dim=0)

class ImageCrossMultiply(ImageMultiply):
	def __init__(self):
		super(ImageCrossMultiply, self).__init__()
	
	def forward(self, volume1, volume2, T):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		mults = []
		
		volume1_unpacked = []
		volume2_unpacked = []
		for i in range(0, num_features):
			volume1_unpacked.append(volume1[:,0:num_features-i,:,:])
			volume2_unpacked.append(volume2[:,i:num_features,:,:])
		volume1 = torch.cat(volume1_unpacked, dim=1)
		volume2 = torch.cat(volume2_unpacked, dim=1)

		for i in range(batch_size):
			v1 = volume1[i,:,:,:].squeeze()
			v2 = volume2[i,:,:,:].squeeze()
			dx = int(T[i,0].item())
			dy = int(T[i,1].item())
			mults.append(self.multiply(v1,v2,dx,dy,volume_size))
		return torch.stack(mults, dim=0)