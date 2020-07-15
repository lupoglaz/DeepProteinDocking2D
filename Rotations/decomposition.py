import os
import sys

import torch
import torch.nn as nn


import numpy as np
import scipy
import math
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

class HarmonicDecomposition(nn.Module):
	def __init__(self, box_size=50, N=20, L=20, alpha=2.0, lam=0.25):
		super(HarmonicDecomposition, self).__init__()
		basis_filename = "N%d_L%d_Al%.2f_Lam%.2f.th"%(N, L, alpha, lam)
		self.N = N
		self.L = L
		self.alpha = alpha
		self.lam = lam
		self.box_size = box_size
		if os.path.exists(basis_filename):
			self.basis = torch.load(basis_filename)
		else:
			self.basis = self.compute_basis(box_size)
			torch.save(self.basis, basis_filename)

	def basis_function(self, box_size, k, l, alpha=2.0, lam=2.0):
		with torch.no_grad():
			re = torch.zeros(box_size, box_size, dtype=torch.float, device='cpu')
			im = torch.zeros(box_size, box_size, dtype=torch.float, device='cpu')
			bsize2 = box_size/2
			max_r = math.sqrt(bsize2*bsize2 + bsize2*bsize2) + 1.0
			
			x = np.linspace(0.0, max_r, 200)
			c = np.exp(-x*lam/(2.0)) * np.power( (x*lam), (alpha-1.0)/2.0)
			norm = np.sqrt(lam*scipy.special.factorial(k)/(scipy.special.gamma(k+1+alpha)))
			radial = c*norm*scipy.special.eval_genlaguerre(k, alpha, x*lam)
			
			for i in range(0, box_size):
				for j in range(0, box_size):
					x = i - bsize2
					y = j - bsize2
					r = math.sqrt(x*x + y*y)
					phi = math.atan2(y, x)
					angular_cos = math.cos(l*phi)/math.sqrt(2.0*math.pi)
					angular_sin = math.sin(l*phi)/math.sqrt(2.0*math.pi)
					re[i,j] = angular_cos * radial[int(200*r/max_r)]
					im[i,j] = angular_sin * radial[int(200*r/max_r)]
			return re, im

	def compute_basis(self, box_size):
		with torch.no_grad():
			basis = []
			for k in range(0, self.N):
				for l in range(0, self.L):
					re, im = self.basis_function(box_size, k, l, self.alpha, self.lam)
					basis.append(re)
					basis.append(im)
			basis = torch.cat(basis, dim=0)
			basis = basis.view(self.N, self.L, 2, box_size, box_size)
			return basis
	
	def euclid2basis(self, x):
		x = x.unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
		basis = self.basis.unsqueeze(dim=0).unsqueeze(dim=1)
		coeffs = torch.sum(torch.sum(x * basis, dim=6), dim=5)
		return coeffs

	def basis2euclid(self, coeffs):
		coeffs = coeffs.unsqueeze(dim=-1).unsqueeze(dim=-1)
		basis = self.basis.unsqueeze(dim=0).unsqueeze(dim=1)
		x = torch.sum(torch.sum(torch.sum(basis * coeffs, dim=4), dim=3), dim=2)
		return x

	def rotate(self, coeffs, angles):
		angles = angles.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
		mult = torch.linspace(0, coeffs.size(3)-1, steps=coeffs.size(3))
		mult = mult.unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
		angles = angles.repeat(1, 1, 1, coeffs.size(3))
		angles = angles*mult
		a = coeffs[:,:,:,:,0] * torch.cos(angles) + coeffs[:,:,:,:,1] * torch.sin(angles)
		b = coeffs[:,:,:,:,1] * torch.cos(angles) - coeffs[:,:,:,:,0] * torch.sin(angles)
		return torch.stack([a, b], dim=4)
		


def plot_harmonics(dec: HarmonicDecomposition):
	f = plt.figure(figsize=(20, 20))
	field = np.zeros( (dec.box_size*dec.N, dec.box_size*dec.L*2) )
	for k in range(0, dec.N):
		for l in range(0, dec.L):
			field[dec.box_size*k:dec.box_size*(k+1), dec.box_size*2*l:dec.box_size*(2*l+1)] = dec.basis[k,l,0,:,:].numpy()
			field[dec.box_size*k:dec.box_size*(k+1), dec.box_size*(2*l+1):dec.box_size*(2*l+2)] = dec.basis[k,l,1,:,:].numpy()

	plt.imshow(field)
	plt.show()

def check_orthogonality(dec: HarmonicDecomposition):
	basis = dec.basis.view(dec.N*dec.L*2, dec.box_size, dec.box_size)
	res = torch.zeros(dec.N*dec.L*2, dec.N*dec.L*2)
	for i in range(dec.N*dec.L*2):
		for j in range(dec.N*dec.L*2):
			res[i,j] = torch.sum(basis[i,:,:]*basis[j,:,:])
	
	for i in range(res.size(0)):
		str = ''
		for j in range(res.size(1)):
			str += '%.1f\t'%res[i,j].item()
		print(str)
		
def check_laguerre_orthogonality(N=10, max_r=25.0, alpha=2.0, lam = 2.0):
	x = np.linspace(0.0, max_r, 100)
	c = np.exp(-x*lam/(2.0)) * np.power( (x*lam), (alpha-1.0)/2.0)
	for i in range(N):
		norm = np.sqrt(lam*scipy.special.factorial(i)/(scipy.special.gamma(i+1+alpha)))
		lag_i = norm*c*scipy.special.eval_genlaguerre(i, alpha, x*lam)
		str = ''
		for j in range(N):
			norm = np.sqrt(lam*scipy.special.factorial(j)/(scipy.special.gamma(j+1+alpha)))
			lag_j = norm*c*scipy.special.eval_genlaguerre(j, alpha, x*lam)
			coef = np.sum(lag_i*lag_j*x)
			str += '%.1f\t'%coef
		print(str)

if __name__=='__main__':
	dec = HarmonicDecomposition(box_size=50, N=30, L=20, alpha=2.0, lam=0.50)
	# plot_harmonics(dec)
	# check_orthogonality(dec)
	
	a = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	gaussian(a, (-10,-10), 2.0)
	a = a.unsqueeze(dim=0).unsqueeze(dim=1)
	b = dec.euclid2basis(a)
	c = dec.basis2euclid(b)
		
	fig = plt.figure()
	camera = Camera(fig)
	for angle in np.linspace(0.0, np.pi, 100):
		angles = torch.tensor([angle], dtype=torch.float32)
		b_rot = dec.rotate(b, angles)
		c_rot = dec.basis2euclid(b_rot)
		plt.imshow(c_rot.squeeze().numpy())	
		camera.snap()
	
	animation = camera.animate()
	animation.save('rotation_animation.mp4')

	# check_laguerre_orthogonality(N=20, max_r=50.0)
	
	plt.subplot(3,1,1)
	plt.imshow(a.squeeze().numpy())
	plt.colorbar()
	plt.subplot(3,1,2)
	plt.imshow(c.squeeze().numpy())
	plt.colorbar()
	plt.subplot(3,1,3)
	plt.imshow(c_rot.squeeze().numpy())
	plt.colorbar()
	plt.tight_layout()
	plt.show()