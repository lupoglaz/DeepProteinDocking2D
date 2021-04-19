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
import shapely.geometry as geom

from tqdm import tqdm

def _pick_translation(receptor, ligand, threshold):
	rec_center = np.array([receptor.shape[0]/2, receptor.shape[1]/2])
	lig_center = np.array([ligand.shape[0]/2, ligand.shape[1]/2])
	
	N_steps = int(receptor.shape[0]/np.sqrt(2.0))
	angle = uniform(0,2.0*np.pi)

	max_overlap = np.sum(receptor*ligand)
	for i in range(0, N_steps): 
		t = np.floor(i*np.array([np.cos(angle), np.sin(angle)]))
		sup_rec, sup_lig = _superpose_volumes(receptor, ligand, t)
		overlap = np.sum(sup_rec*sup_lig)/max_overlap
		if overlap<threshold:
			return t, overlap

	return None, None
	
def _superpose_volumes(v1, v2, translation):
	def mabs(dx, dy):		
		return np.abs(dx), np.abs(dy)
	dx = int(translation[0])
	dy = int(translation[1])
	
	if v1.shape[0]!=v1.shape[1]:
		raise Exception("v1 shapes not equal")
	if v2.shape[0]!=v2.shape[1]:
		raise Exception("v2 shapes not equal")
	if v1.shape[0]!=v2.shape[0]:
		raise Exception("v1 and v2 shapes not equal")
	
	L = v1.shape[0]
	v1_crop = None
	v2_crop = None
	#all positive
	if dx>=0 and dy>=0:
		dx, dy = mabs(dx, dy)
		v1_crop = v1[dx:L, dy:L]
		v2_crop = v2[0:L-dx, 0:L-dy]
		
	#one negative
	elif dx<0 and dy>=0:
		dx, dy= mabs(dx, dy)
		v1_crop = v1[0:L-dx, dy:L]
		v2_crop = v2[dx:L, 0:L-dy]
	elif dx>=0 and dy<0:
		dx, dy = mabs(dx, dy)
		v1_crop = v1[dx:L, 0:L-dy]
		v2_crop = v2[0:L-dx, dy:L]
		
	#all negative
	elif dx<0 and dy<0:
		dx, dy = mabs(dx, dy)
		v1_crop = v1[0:L-dx, 0:L-dy]
		v2_crop = v2[dx:L, dy:L]
	
	return v1_crop, v2_crop

class Complex:
	def __init__(self, receptor, ligand, rotation, translation):
		self.receptor = receptor
		self.ligand = ligand
		self.rotation = rotation
		self.translation = translation

	@classmethod
	def generate(cls, receptor, ligand, threshold=0.3, max_iter=100):
		for attempt in range(max_iter):
			rotation = uniform(-np.pi, np.pi)
			rligand = ligand.rotate(rotation)
			translation, overlap = _pick_translation(receptor.bulk, rligand.bulk, threshold)
			if not translation is None:
				if not(ligand.hull is None):
					trligand = rligand.translate(translation)
					poly = receptor.hull.difference(trligand.hull)
					if poly.geom_type != 'Polygon':
						translation = None
						continue
				break
			
		rec, lig = _superpose_volumes(receptor.bulk, rligand.bulk, translation)
		rec[ lig>0.05 ] = 0
		
		return cls(receptor, ligand, rotation, translation)

	def score(self, boundary_size=3, a00=1.0, a11=-1.0, a10=-1.0):
		self.receptor.make_boundary(boundary_size=boundary_size)
		self.ligand.make_boundary(boundary_size=boundary_size)
		trligand = self.ligand.rotate(self.rotation).translate(self.translation)
		t00 = np.sum(self.receptor.bulk * trligand.bulk)
		t11 = np.sum(self.receptor.boundary * trligand.boundary)
		t10 = np.sum(self.receptor.bulk * trligand.boundary + self.receptor.boundary * trligand.bulk)
		score = a11*a11 + a10*t10 + a00*t00
		return score
	
	def get_canvas(self, cell_size):
		rligand = self.ligand.rotate(self.rotation)
		trligand = rligand.translate(self.translation)
		dx, dy = int(self.translation[0]), int(self.translation[1])
		prot_size = self.receptor.bulk.shape[0]

		rec_x_start = int(cell_size/2) - int(prot_size/2) - int(dx/2)
		rec_x_end = int(cell_size/2) + int(prot_size/2) - int(dx/2)
		rec_y_start = int(cell_size/2) - int(prot_size/2) - int(dy/2)
		rec_y_end = int(cell_size/2) + int(prot_size/2) - int(dy/2)
		lig_x_start = int(cell_size/2) - int(prot_size/2) + int(dx/2)
		lig_x_end = int(cell_size/2) + int(prot_size/2) + int(dx/2)
		lig_y_start = int(cell_size/2) - int(prot_size/2) + int(dy/2)
		lig_y_end = int(cell_size/2) + int(prot_size/2) + int(dy/2)

		canvas = np.zeros( (cell_size, cell_size) )
		canvas[rec_x_start:rec_x_end, rec_y_start:rec_y_end] = self.receptor.bulk
		canvas[lig_x_start:lig_x_end, lig_y_start:lig_y_end] += 2*rligand.bulk
		return canvas

	def plot(self):
		rligand = self.ligand.rotate(self.rotation)
		trligand = rligand.translate(self.translation)
		dx, dy = int(self.translation[0]), int(self.translation[1])
		# print('Translation=', dx, dy, 'Rotation:', self.rotation)
		# print('Score=', self.score())
		field = np.zeros( (200, 200) )
		field[75:125, 75:125] = self.receptor.bulk
		field[  100 + dx - 25: 100 + dx + 25, 
				100 + dy - 25: 100 + dy + 25] += 2*rligand.bulk
		# field[  75: 125, 
		#  		75: 125] += 2*trligand.bulk
		field[150:200,150:200]  = self.receptor.bulk
		field[150:200,100:150]  = self.ligand.bulk
		plt.imshow(field)
		# plt.show()

def scan_parameter(param, func, output_name='gap_score_param1.png', num_samples=10, name=""):
	f = plt.figure(figsize=(num_samples,len(param)+0.5))
	cell_size = 80
	canvas = np.zeros((cell_size*num_samples, cell_size*len(param)))
	plot_num = 0 
	for i, p in tqdm(enumerate(param), total=len(param)):
		for j in range(num_samples):
			rec = Protein.generateConcave(size=50, num_points=80, alpha=0.90)
			lig = Protein.generateConcave(size=50, num_points=80, alpha=0.90)
			cplx = func(rec, lig, p)
			canvas[j*cell_size:(j+1)*cell_size, i*cell_size:(i+1)*cell_size] = cplx.get_canvas(cell_size)
			
	plt.imshow(canvas.transpose(), origin='lower', interpolation='nearest', resample=False, filternorm=False)
	plt.xticks(ticks=[i*cell_size + cell_size/2 for i in range(num_samples)], labels=['%d'%(i+1) for i in range(num_samples)])
	plt.yticks(ticks=[i*cell_size + cell_size/2 for i in range(len(param))], labels=['%.2f'%i for i in param])
	plt.ylabel(name)
	plt.xlabel('sample number')
	plt.tight_layout()
	plt.savefig(output_name)



if __name__=='__main__':
	rec = Protein.generateConcave(size=50, num_points=50)
	lig = Protein.generateConcave(size=50, num_points=50)
	cplx = Complex.generate(rec, lig)
	cplx.plot()
	plt.show()

	# scan_parameter(param=np.arange(0.1,0.75,0.05, dtype=np.float32), 
	# 				func=lambda x, y, p: Complex.generate(x, y, threshold=p),
	# 				num_samples=10, 
	# 				output_name='comp_overlap.png', name='Overlap')