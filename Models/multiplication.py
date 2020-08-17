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
		
		vol = torch.zeros_like(v1)
		#all positive
		if dx>=0 and dy>=0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,dx:L, dy:L] * v2[:,0:L-dx, 0:L-dy]
			vol[:,dx:L, dy:L] = v2[:,0:L-dx, 0:L-dy]
		
		#one negative
		elif dx<0 and dy>=0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,0:L-dx, dy:L] * v2[:,dx:L, 0:L-dy]
			vol[:,0:L-dx, dy:L] = v2[:,dx:L, 0:L-dy]

		elif dx>=0 and dy<0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,dx:L, 0:L-dy] * v2[:,0:L-dx, dy:L]
			vol[:,dx:L, 0:L-dy] = v2[:,0:L-dx, dy:L]
		
		#all negative
		elif dx<0 and dy<0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:,0:L-dx, 0:L-dy] * v2[:,dx:L, dy:L]
			vol[:,0:L-dx, 0:L-dy] = v2[:,dx:L, dy:L]

		return result.sum(dim=2).sum(dim=1).squeeze(), vol

	def forward(self, receptor, ligand, T):
		batch_size = receptor.size(0)
		L = receptor.size(2)
		mults = []
		vols = []
		for i in range(batch_size):
			v1 = receptor[i,:,:,:]
			v2 = ligand[i,:,:,:]
			dx = int(T[i,0])
			dy = int(T[i,1])
			res, vol = self.multiply(v1,v2,dx,dy,L)
			mults.append(res)
			vols.append(vol)
		return torch.stack(mults, dim=0), torch.stack(vols, dim=0)

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
			v1 = volume1[i,:,:,:]
			v2 = volume2[i,:,:,:]
			dx = int(T[i,0].item())
			dy = int(T[i,1].item())
			mults.append(self.multiply(v1,v2,dx,dy,volume_size))
		return torch.stack(mults, dim=0)

def test_translation(a,b):
	fig = plt.figure(figsize=(20,5))
	camera = Camera(fig)
	mult = ImageMultiply()
	for i in range(50):
		dr = torch.tensor([[0.0, float(i)-25.0]], dtype=torch.float, device='cpu')
		m, v2 = mult(b, a, dr)

		plt.subplot(1,3,1)
		plt.imshow(a.squeeze())
		plt.subplot(1,3,2)
		plt.imshow(b.squeeze())
		plt.subplot(1,3,3)
		plt.imshow(v2.detach().squeeze())
		camera.snap()
	animation = camera.animate()
	plt.show()

if __name__=='__main__':
	import seaborn
	from matplotlib import pylab as plt
	from celluloid import Camera
	def gaussian(tensor, center=(0,0), sigma=1.0):
		center = center[0] + tensor.size(0)/2, center[1] + tensor.size(1)/2
		for x in range(tensor.size(0)):
			for y in range(tensor.size(1)):
				r2 = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])
				arg = torch.tensor([-r2/sigma])
				tensor[x,y] = torch.exp(arg)
	
	
	a = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	b = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	t = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	gaussian(a, (0,0), 3.0)
	gaussian(t, (4.9,0), 3.0)
	a-=t
	gaussian(b, (10,0), 3.0)    
	gaussian(t, (10,5.1), 3.0)    
	b-=t
	a = a.unsqueeze(dim=0).unsqueeze(dim=1)
	b = b.unsqueeze(dim=0).unsqueeze(dim=1)

	# test_optimization(a,b)
	test_translation(a,b)