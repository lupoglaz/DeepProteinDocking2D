import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from collections import namedtuple
from random import uniform
import scipy
from scipy.spatial import ConvexHull
from scipy import ndimage
import alphashape
import shapely.geometry as geom
import shapely.affinity as affine
import descartes.patch as patch



def get_random_points(num_points, xspan, yspan):
	points = [[uniform(*xspan), uniform(*yspan)]  for i in range(num_points)]
	radius_x = (xspan[1] - xspan[0])/2.0
	radius_y = (yspan[1] - yspan[0])/2.0
	center_x = (xspan[1] + xspan[0])/2.0
	center_y = (yspan[1] + yspan[0])/2.0
	points = [[x,y] for x,y in points if np.sqrt(((x-center_x)/radius_x) ** 2 + ((y-center_y)/radius_y) ** 2) < 1.0]
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
			inside = False
			if isinstance(hull, ConvexHull):
				inside = is_p_inside_points_hull(hull, np.array([[x,y]]))
			elif isinstance(hull, geom.Polygon):
				inside = geom.Point(x,y).within(hull)
			if inside:
				array[i, j] = 1.0
			else:
				array[i, j] = 0.0
	return array

class Protein:

	def __init__(self, bulk, boundary=None, hull=None):
		self.size = bulk.shape[0]
		self.bulk = bulk
		self.boundary = boundary
		self.hull = hull
	
	@classmethod
	def generateConvex(cls, size=50, num_points = 15, occupancy=0.8):
		grid_coordinate_span = (0, size)
		points_coordinate_span = (0.5*(1.0-occupancy)*size, size - 0.5*(1.0-occupancy)*size)
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		hull = scipy.spatial.ConvexHull(points)		
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk)

	@classmethod
	def generateConcave(cls, size=50, alpha=1.0, num_points = 25, occupancy=0.8):
		grid_coordinate_span = (0, size)
		points_coordinate_span = (0.5*(1.0-occupancy)*size, size - 0.5*(1.0-occupancy)*size)
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		optimal = alphashape.optimizealpha(points)
		hull = alphashape.alphashape(points, optimal)
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk, hull=hull)

	def rotate(self, rotation_angle):
		blk = ndimage.rotate(self.bulk, 180.0*rotation_angle/np.pi, reshape=False, mode='constant', cval=0.0, prefilter=False)
		
		hull = None
		if not (self.hull is None):
			hull = affine.rotate(self.hull, 180.0*rotation_angle/np.pi, origin=(self.size/2.0, self.size/2.0))
		
		bnd = None	
		if not (self.boundary is None):
			bnd = ndimage.rotate(self.boundary, 180.0*rotation_angle/np.pi, reshape=False, mode='constant', cval=0.0, prefilter=False)
		
		return Protein(bulk=blk, boundary=bnd, hull=hull)

	def translate(self, translation):
		blk = ndimage.shift(self.bulk, translation, prefilter=False, mode='constant', cval=0.0)
		
		hull = None
		if not(self.hull is None):
			hull = affine.translate(self.hull, translation[0], translation[1])

		bnd = None
		if not(self.boundary is None):
			bnd = ndimage.shift(self.boundary, translation, prefilter=False, mode='constant', cval=0.0)
		
		return Protein(bulk=blk, boundary=bnd, hull=hull)


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
		return plt.imshow(self.bulk.transpose(), origin='lower', interpolation='nearest', resample=False, filternorm=False)
	
	def plot_hull(self):
		plt.plot(*(self.hull.exterior.xy))

def test_protein_generation():
	N_prot = 5
	field = np.zeros((50,N_prot*50))
	for i in range(N_prot):
		# prot = Protein.generateConvex(size=50)
		prot = Protein.generateConcave(size=50)
		field[:, i*50:(i+1)*50] = prot.bulk

	plt.imshow(field, origin='lower')
	plt.show()

def test_rotations():
	Nrot = 10
	ligand = Protein.generateConcave(size=50)
	field = np.zeros((50, 50+Nrot*50))
	field[:,:50] = ligand.bulk
	for i in range(Nrot):
		# rotation_angle = uniform(0, 360)
		rotation_angle = (i+1)*360.0/Nrot
		rligand = ligand.rotate(np.pi*rotation_angle/180.0)        
		field[:,(i+1)*50:(i+2)*50] = rligand.bulk
	
	plt.imshow(field)
	plt.show()

def test_translations():
	Ntr = 10
	ligand = Protein.generateConcave(size=50)
	field = np.zeros((50, 50+Ntr*50))
	field[:,:50] = ligand.bulk
	for i in range(Ntr):
		translation = np.array([0+i*2, 0])
		rligand = ligand.translate(translation)        
		field[:,(i+1)*50:(i+2)*50] = rligand.bulk
	
	plt.imshow(field)
	plt.show()

def test_representation():
	prot = Protein.generateConcave()
	prot.make_boundary()
	features = prot.get_repr()
	
	field = np.zeros( (50, 100) )
	field[:, 0:50] = features[0,:,:]
	field[:, 50:100] = features[1,:,:]
	plt.imshow(field)
	plt.show()

def test_hull():
	prot_init = Protein.generateConcave(size=50,num_points=70)
	plt.subplot(221)
	prot_init.plot_bulk()
	prot_init.plot_hull()

	prot = prot_init.rotate(0.6)
	plt.subplot(222)
	prot.plot_bulk()
	prot.plot_hull()

	prot = prot_init.translate(np.array([10,10]))
	plt.subplot(223)
	prot.plot_bulk()
	prot.plot_hull()
	
	prot = prot_init.rotate(0.6).translate(np.array([10,10]))
	plt.subplot(224)
	prot.plot_bulk()
	prot.plot_hull()
	
	plt.show()

if __name__ == '__main__':
	test_hull()
		
	# test_representation()
	# test_translations()

	# test_protein_generation()
	# test_rotations()
	# test_translations()