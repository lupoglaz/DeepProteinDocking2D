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
import scipy
from scipy.spatial import ConvexHull
from scipy import ndimage

def get_random_points(num_points, xspan, yspan):
	points = [[uniform(*xspan), uniform(*yspan)]  for i in range(num_points)]
	points = np.array(points)
	return points

def is_p_inside_points_hull(hull, p):
	new_points = np.append(hull.points, p, axis=0)
	new_hull = ConvexHull(new_points)
	if list(hull.vertices) == list(new_hull.vertices):
		return True
	else:
		return False

def hull2array(hull, array, xspan, yspan):
	x_size = array.shape[0]
	y_size = array.shape[1]
	for i in range(x_size):
		for j in range(y_size):
			x = xspan[0] + i*float(xspan[1]-xspan[0])/float(x_size)
			y = yspan[0] + j*float(yspan[1]-yspan[0])/float(y_size)
			if is_p_inside_points_hull(hull, np.array([[x,y]])):
				array[i, j] = 1.0
			else:
				array[i, j] = 0.0
	return array

class Protein:
	def __init__(self, bulk, boundary=None):
		self.size = bulk.shape[0]
		self.bulk = bulk
		self.boundary = boundary
	
	@classmethod
	def generate(cls, size=50, grid_coordinate_span = (0,10), points_coordinate_span = (1,9), num_points = 15):		
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		hull = scipy.spatial.ConvexHull(points)
				
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		boundary = None
		return cls(bulk, boundary)

	def rotate(self, rotation_angle):
		if self.boundary is None:
			blk = ndimage.rotate(self.bulk, 180.0*rotation_angle/np.pi, reshape=False, mode='nearest', cval=0.0)
			return Protein(bulk=blk)
		else:
			blk = ndimage.rotate(self.bulk, 180.0*rotation_angle/np.pi, reshape=False, mode='nearest', cval=0.0)
			bnd = ndimage.rotate(self.boundary, 180.0*rotation_angle/np.pi, reshape=False, mode='nearest', cval=0.0)
			return Protein(bulk=blk, boundary=bnd)

	def translate(self, translation):
		if self.boundary is None:
			blk = ndimage.shift(self.bulk, translation, prefilter=False)
			return Protein(bulk=blk)
		else:
			blk = ndimage.shift(self.bulk, translation, prefilter=False)
			bnd = ndimage.shift(self.boundary, translation, prefilter=False)
			return Protein(bulk=blk, boundary=bnd)


	def make_boundary(self, boundary_size=3):
		self.boundary = np.zeros((self.size, self.size))
		for i in range(self.size):
			for j in range(self.size):
				if self.bulk[i,j]>0.5:
					continue
				x_low = max(i - boundary_size, 0)
				x_high = min(i + boundary_size, self.size)
				y_low = max(j - boundary_size, 0)
				y_high = min(j + boundary_size, self.size)
				if np.sum(self.bulk[x_low:x_high, y_low:y_high])>0.5:
					self.boundary[i,j]=1.0
	
	def get_repr(self):
		if self.boundary is None:
			features = self.bulk
		else:
			features = np.stack([self.bulk, self.boundary], axis=0)
		return features
		
	def plot_bulk(self):
		plt.imshow(self.bulk.transpose(), origin='lower')

def test_protein_generation():
	N_prot = 5
	field = np.zeros((50,N_prot*50))
	for i in range(N_prot):
		prot = Protein.generate(size=50)
		field[:, i*50:(i+1)*50] = prot.bulk

	plt.imshow(field, origin='lower')
	plt.show()

def test_rotations():
	Nrot = 10
	ligand = Protein.generate(size=50)
	field = np.zeros((50, 50+Nrot*50))
	field[:,:50] = ligand.bulk
	for i in range(Nrot):
		rotation_angle = uniform(0, 360)
		rligand = ligand.rotate(np.pi*rotation_angle/180.0)        
		field[:,(i+1)*50:(i+2)*50] = rligand.bulk
	
	plt.imshow(field)
	plt.show()

def test_translations():
	Ntr = 10
	ligand = Protein.generate(size=50)
	field = np.zeros((50, 50+Ntr*50))
	field[:,:50] = ligand.bulk
	for i in range(Ntr):
		translation = np.array([0+i*2, 0])
		rligand = ligand.translate(translation)        
		field[:,(i+1)*50:(i+2)*50] = rligand.bulk
	
	plt.imshow(field)
	plt.show()

def test_representation():
	prot = Protein.generate()
	prot.make_boundary()
	features = prot.get_repr()
	
	field = np.zeros( (50, 100) )
	field[:, 0:50] = features[0,:,:]
	field[:, 50:100] = features[1,:,:]
	plt.imshow(field)
	plt.show()

if __name__ == '__main__':
	prot = Protein.generate(size=50)
	# prot.plot_bulk()
	# plt.show()
	# test_representation()
	test_translations()

	#test_protein_generation()
	#test_rotations()