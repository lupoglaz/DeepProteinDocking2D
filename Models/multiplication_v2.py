import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from torch import optim

class ImageCrossMultiplyV2(nn.Module):
	def __init__(self):
		super(ImageCrossMultiplyV2, self).__init__()
	
	def multiply(self, v1, v2, dx, dy, L):
		def mabs(self, dx, dy):
			return np.abs(dx), np.abs(dy)
		
		#all positive
		if dx>=0 and dy>=0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:, dx:L, dy:L] * v2[:, 0:L-dx, 0:L-dy]
		
		#one negative
		elif dx<0 and dy>=0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:, 0:L-dx, dy:L] * v2[:, dx:L, 0:L-dy]
		elif dx>=0 and dy<0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:, dx:L, 0:L-dy] * v2[:, 0:L-dx, dy:L]
		
		#all negative
		elif dx<0 and dy<0:
			dx, dy = self.mabs(dx, dy)
			result = v1[:, 0:L-dx, 0:L-dy] * v2[:,dx:L, dy:L]

		return result.sum(dim=2).sum(dim=1).squeeze()

	def forward(self, volume1, volume2, alpha, dr):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		mults = []
		
		T0 = torch.cat([torch.cos(alpha), -torch.sin(alpha)], dim=1)
		T1 = torch.cat([torch.sin(alpha), torch.cos(alpha)], dim=1)
		t = torch.stack([(T0*dr).sum(dim=1), (T1*dr).sum(dim=1)], dim=1)
		T01 = torch.stack([T0, T1], dim=1)
		A = torch.cat([T01, t.unsqueeze(dim=2)], dim=2)
		
		grid = nn.functional.affine_grid(A, size=volume2.size())
		volume2 = nn.functional.grid_sample(volume2, grid)
		volume1_unpacked = []
		volume2_unpacked = []
		for i in range(0, num_features):
			volume1_unpacked.append(volume1[:,0:num_features-i,:,:])
			volume2_unpacked.append(volume2[:,i:num_features,:,:])
		volume1 = torch.cat(volume1_unpacked, dim=1)
		volume2 = torch.cat(volume2_unpacked, dim=1)

		mults = (volume1 * volume2).sum(dim=3).sum(dim=2)
		
		return mults, volume2, grid

def test_optimization(a,b):
	mult = ImageCrossMultiplyV2()
	alpha = torch.tensor([[-0.6]], dtype=torch.float, device='cpu').requires_grad_()
	dr = torch.tensor([[0.0, 0.0]], dtype=torch.float, device='cpu').requires_grad_()
	
	fig = plt.figure(figsize=(20,5))
	camera = Camera(fig)

	optimizer = optim.Adam([alpha, dr], lr = 0.1)
	for i in range(100):
		optimizer.zero_grad()		
		m, v2 = mult(a, b, alpha, dr)
		loss = -m.sum()
		loss.backward()
		optimizer.step()
		plt.subplot(1,3,1)
		plt.imshow(a.squeeze())
		plt.subplot(1,3,2)
		plt.imshow(b.squeeze())
		plt.subplot(1,3,3)
		plt.imshow(v2.detach().squeeze(), label=f'step {i}')
		camera.snap()

	animation = camera.animate()
	plt.show()

def test_translation(a,b):
	fig = plt.figure(figsize=(20,5))
	camera = Camera(fig)
	mult = ImageCrossMultiplyV2()
	for i in range(100):
		alpha = torch.tensor([[2*np.pi*float(i)/100.0]], dtype=torch.float, device='cpu')
		dr = torch.tensor([[float(i)/100.0, 0.0]], dtype=torch.float, device='cpu')
		m, v2 = mult(b, a, alpha, dr)

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

	test_optimization(a,b)
	
	