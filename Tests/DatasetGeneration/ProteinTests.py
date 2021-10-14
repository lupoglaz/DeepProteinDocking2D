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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from DatasetGeneration import Protein

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

def test_rmsd():
	prot_init = Protein.generateConcave(size=50,num_points=70)
	num_angles = 360
	angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=num_angles))
	rmsd = prot_init.grid_rmsd(angles, [10,5], angles[190])
	plt.subplot(221)
	prot_init.plot_bulk()

	angle_idx = 200
	translation = [10,0]
	prot = prot_init.rotate(angles[angle_idx]).translate(translation)
	plt.subplot(222)
	prot.plot_bulk()
	
	plt.subplot(223)
	plt.imshow(rmsd[angle_idx,:,:])
	plt.title(f'rmsd = {rmsd[angle_idx, translation[0]+50, translation[1]+50].item()}')

	plt.show()


if __name__ == '__main__':
	# test_hull()
	# test_protein_generation()
		
	# test_representation()
	# test_translations()
	test_rotations()
	
	# test_rmsd()
	