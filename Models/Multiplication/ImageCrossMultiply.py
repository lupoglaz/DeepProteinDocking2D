import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from torch import optim

class ImageCrossMultiply(nn.Module):
	def __init__(self, full=True):
		super(ImageCrossMultiply, self).__init__()
		self.full = full

	def forward(self, volume1, volume2, alpha, dr):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		mults = []
		perm = torch.tensor([1,0], dtype=torch.long, device=volume1.device)
		dr = -2.0*dr[:, perm]/volume_size
		T0 = torch.cat([torch.cos(alpha), -torch.sin(alpha)], dim=1)
		T1 = torch.cat([torch.sin(alpha), torch.cos(alpha)], dim=1)
		t = torch.stack([(T0*dr).sum(dim=1), (T1*dr).sum(dim=1)], dim=1)
		T01 = torch.stack([T0, T1], dim=1)
		A = torch.cat([T01, t.unsqueeze(dim=2)], dim=2)
		
		grid = nn.functional.affine_grid(A, size=volume2.size())
		volume2 = nn.functional.grid_sample(volume2, grid)
		if not self.full:
			volume1_unpacked = []
			volume2_unpacked = []
			for i in range(0, num_features):
				volume1_unpacked.append(volume1[:,0:num_features-i,:,:])
				volume2_unpacked.append(volume2[:,i:num_features,:,:])
			volume1 = torch.cat(volume1_unpacked, dim=1)
			volume2 = torch.cat(volume2_unpacked, dim=1)
		else:
			volume1 = volume1.unsqueeze(dim=2).repeat(1, 1, num_features, 1, 1)
			volume2 = volume2.unsqueeze(dim=1).repeat(1, num_features, 1, 1, 1)
			volume1 = volume1.view(batch_size, num_features*num_features, volume_size, volume_size)
			volume2 = volume2.view(batch_size, num_features*num_features, volume_size, volume_size)
			

		mults = (volume1 * volume2).sum(dim=3).sum(dim=2)
		
		return mults, volume2, grid

def test_optimization(a,b):
	mult = ImageCrossMultiply()
	alpha = torch.tensor([[-0.6]], dtype=torch.float, device='cpu').requires_grad_()
	dr = torch.tensor([[0.0, 0.0]], dtype=torch.float, device='cpu').requires_grad_()
	
	fig = plt.figure(figsize=(20,5))
	camera = Camera(fig)

	optimizer = optim.Adam([alpha, dr], lr = 0.1)
	for i in range(100):
		optimizer.zero_grad()		
		m, v2, _ = mult(a, b, alpha, dr)
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
	mult = ImageCrossMultiply()
	for i in range(50):
		alpha = torch.tensor([[2*np.pi*float(i)/100.0]], dtype=torch.float, device='cpu')
		dr = torch.tensor([[0.0, float(i)-25.0]], dtype=torch.float, device='cpu')
		m, v2, _ = mult(b, a, alpha, dr)

		plt.subplot(1,3,1)
		plt.imshow(a.squeeze())
		plt.subplot(1,3,2)
		plt.imshow(b.squeeze())
		plt.subplot(1,3,3)
		plt.imshow(v2.detach().squeeze())
		camera.snap()
	animation = camera.animate()
	plt.show()

def test_mult_order():
	mult = ImageCrossMultiply()
	alpha = torch.tensor([[2*np.pi]], dtype=torch.float, device='cpu')
	dr = torch.tensor([[0.0, 0.0]], dtype=torch.float, device='cpu')
	a = torch.zeros(1, 2, 50, 50, dtype=torch.float, device='cpu')
	a[0,0,:,:] = 1.0
	a[0,1,:,:] = 2.0
	b = torch.zeros(1, 2, 50, 50, dtype=torch.float, device='cpu')
	b[0,0,:,:] = 3.0
	b[0,1,:,:] = 4.0
	m, v2, _ = mult(a,b, alpha, dr)
	for k in range(m.size(1)):
		print(k, m[0, k]/(50*50))



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
	# test_translation(a,b)
	test_mult_order()
	
	