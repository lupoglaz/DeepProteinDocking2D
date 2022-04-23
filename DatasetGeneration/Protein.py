import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl
import math

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

import torch.nn.functional as F

from tqdm import tqdm


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
		optimal = alpha * alphashape.optimizealpha(points)
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

	def make_boundary(self, boundary_size=None):
		epsilon = 1e-5
		sobel_top = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.double)
		sobel_left = sobel_top.t()

		feat_top = F.conv2d(torch.from_numpy(self.bulk).unsqueeze(0).unsqueeze(0), weight=sobel_top.unsqueeze(0).unsqueeze(0), padding=1)
		feat_left = F.conv2d(torch.from_numpy(self.bulk).unsqueeze(0).unsqueeze(0), weight=sobel_left.unsqueeze(0).unsqueeze(0),padding=1)

		top = feat_top.squeeze() + epsilon
		right = feat_left.squeeze() + epsilon
		self.boundary = torch.sqrt(top ** 2 + right ** 2).numpy()


	def get_XC(self):
		"""
		Analog of inertia tensor and center of mass for rmsd calc. Return 2/W * X and C
		"""
		X = torch.zeros(2,2)
		C = torch.zeros(2)
		x_i = (torch.arange(self.size).unsqueeze(dim=0) - self.size/2.0).repeat(self.size, 1)
		y_i = (torch.arange(self.size).unsqueeze(dim=1) - self.size/2.0).repeat(1, self.size)
		mask = torch.from_numpy(self.bulk > 0.5)
		W = torch.sum(mask.to(dtype=torch.float32)) + 1E-5
		x_i = x_i.masked_select(mask)
		y_i = y_i.masked_select(mask)
		#Inertia tensor
		X[0,0] = torch.sum(x_i*x_i)
		X[1,1] = torch.sum(y_i*y_i)
		X[0,1] = torch.sum(x_i*y_i)
		X[1,0] = torch.sum(y_i*x_i)
		#Center of mass
		C[0] = torch.sum(x_i)
		C[1] = torch.sum(x_i)
		return 2.0*X/W, C/W

	def grid_rmsd(self, angles, translation=None, angle=None):
		num_angles = angles.size(0)
		
		X, C = self.get_XC()
		X = X.unsqueeze(dim=0)
		C = C.unsqueeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
		
		#Translations
		T = torch.zeros(self.size*2, self.size*2, 2)
		T[:,:,1] = (torch.arange(self.size*2).unsqueeze(dim=0) - self.size).repeat(self.size*2, 1)
		T[:,:,0] = (torch.arange(self.size*2).unsqueeze(dim=1) - self.size).repeat(1, self.size*2)
		if not(translation is None):
			# T -> T1 - T2
			T[:,:,0] -= translation[0]
			T[:,:,1] -= translation[1]

		T = T.unsqueeze(dim=0).repeat(num_angles, 1, 1, 1)
		
		#Rotations
		R = torch.zeros(num_angles, 2, 2)
		R[:,0,0] = torch.cos(angles)
		R[:,1,1] = torch.cos(angles)
		R[:,1,0] = torch.sin(angles)
		R[:,0,1] = -torch.sin(angles)
		Rtot = R.unsqueeze(dim=1).unsqueeze(dim=2)
		I = torch.diag(torch.ones(2)).unsqueeze(dim=0)
		Itot = I.unsqueeze(dim=1).unsqueeze(dim=2)
		if not(angle is None):
			# Itot -> R2
			Itot = torch.zeros(1, 2, 2)
			Itot[0,0,0] = torch.cos(angle)
			Itot[0,1,1] = torch.cos(angle)
			Itot[0,1,0] = torch.sin(angle)
			Itot[0,0,1] = -torch.sin(angle)
			# R -> R2.T @ R1
			R = Itot.transpose(1,2) @ R
			Itot = Itot.unsqueeze(dim=1).unsqueeze(dim=2)
			
		#RMSD
		rmsd = torch.sum(T*T, dim=3)
		rmsd = rmsd + torch.sum((I-R)*X, dim=(1,2)).unsqueeze(dim=1).unsqueeze(dim=2)
		rmsd = rmsd + 2.0*torch.sum(torch.sum(T.unsqueeze(dim=4) * (Rtot-Itot), dim=3) * C, dim=3)
		return torch.sqrt(rmsd)

	def rmsd(self, translation1, rotation1, translation2, rotation2):
		X, C = self.get_XC()
		T1 = torch.tensor(translation1)
		T2 = torch.tensor(translation2)
		T = T1 - T2
		
		rotation1 = torch.tensor([rotation1])
		rotation2 = torch.tensor([rotation2])
		R1 = torch.zeros(2, 2)
		R1[0,0] = torch.cos(rotation1)
		R1[1,1] = torch.cos(rotation1)
		R1[1,0] = torch.sin(rotation1)
		R1[0,1] = -torch.sin(rotation1)
		R2 = torch.zeros(2, 2)
		R2[0,0] = torch.cos(rotation2)
		R2[1,1] = torch.cos(rotation2)
		R2[1,0] = torch.sin(rotation2)
		R2[0,1] = -torch.sin(rotation2)
		R = R2.transpose(0,1) @ R1
		
		I = torch.diag(torch.ones(2))
		#RMSD
		rmsd = torch.sum(T*T)
		rmsd = rmsd + torch.sum((I-R)*X, dim=(0,1))
		rmsd = rmsd + 2.0*torch.sum(torch.sum(T.unsqueeze(dim=1) * (R1-R2), dim=0) * C, dim=0)
		if rmsd < 0:
			rmsd = 0.0
		else:
			rmsd = torch.sqrt(rmsd)
		return rmsd
	
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
