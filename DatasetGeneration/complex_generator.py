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
from .prot_generator import generate_protein

from scipy import ndimage

def mabs(dx, dy):		
	return np.abs(dx), np.abs(dy)

def rotate_ligand(ligand, rotation_angle):
	ligand = ndimage.rotate(ligand, rotation_angle, reshape=False, mode='nearest', cval=0.0)
	return ligand

def test_rotations():
	Nrot = 5
	ligand = generate_protein(size=50)
	field = np.zeros((50, 50+Nrot*50))
	field[:,:50] = ligand
	for i in range(Nrot):
		rotation_angle = uniform(0, 360)
		rligand = rotate_ligand(ligand, rotation_angle)        
		field[:,(i+1)*50:(i+2)*50] = rligand
	
	plt.imshow(field)
	plt.show()


def superpose_volumes(v1, v2, translation):
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

def test_superpos():
	receptor = generate_protein(size=50)
	ligand = generate_protein(size=50)
	field = np.zeros((50, 200))
	field[:,:50] = receptor
	field[:,50:100] = ligand
	rec_crop, lig_crop = superpose_volumes(receptor, ligand, [20,20])
	rec_crop = np.pad(  rec_crop, 
						((0,50-rec_crop.shape[0]),
						(0,50-rec_crop.shape[1])), 
						mode='constant',
						constant_values=((0, 0), (0,0)))
	lig_crop = np.pad(  lig_crop, 
						((0,50-lig_crop.shape[0]),
						(0,50-lig_crop.shape[1])), 
						mode='constant',
						constant_values=((0, 0), (0,0)))
	field[:,100:150] = rec_crop
	field[:,150:200] = lig_crop

	plt.imshow(field)
	plt.show()

def pick_translation(receptor, ligand, threshold=0.2, debug=False):
	rec_center = np.array([receptor.shape[0]/2, receptor.shape[1]/2])
	lig_center = np.array([ligand.shape[0]/2, ligand.shape[1]/2])
	
	N_steps = int(receptor.shape[0]/np.sqrt(2.0))
	angle = uniform(0,2.0*np.pi)

	if debug:    
		overlaps = []

	max_overlap = np.sum(receptor*ligand)
	for i in range(0, N_steps): 
		t = np.floor(i*np.array([np.cos(angle), np.sin(angle)]))
		sup_rec, sup_lig = superpose_volumes(receptor, ligand, t)
		overlap = np.sum(sup_rec*sup_lig)/max_overlap
		if overlap<threshold:
			return t, overlap
		
		if debug:
			overlaps.append(overlap)

	if debug:
		plt.plot(overlaps)
		plt.show()
	
	return None, None

def test_translation_picking():
	receptor = generate_protein(size=50)
	ligand = generate_protein(size=50)
	t, ovelap = pick_translation(receptor, ligand, threshold=0.1, debug=True)
	dx, dy = int(t[0]), int(t[1])
	field = np.zeros( (200, 200) )
	field[75:125, 75:125] = receptor
	print('Translation=', dx, dy)
	field[  100 + dx - 25: 100 + dx + 25, 
			100 + dy - 25: 100 + dy + 25] += ligand
	plt.imshow(field)
	plt.show()

def generate_complex(protein_size=50, threshold_correct=0.3, threshold_incorrect=0.2, num_rotations=5):
	receptor = generate_protein(size=protein_size)
	ligand = generate_protein(size=protein_size)

	rotation_angle_correct = uniform(0, 360)
	rligand = rotate_ligand(ligand, rotation_angle_correct)
	t_correct, overlap = pick_translation(receptor, rligand, threshold=threshold_correct, debug=False)
	if t_correct is None:
		return receptor, ligand, None, None
	sup_rec, sup_lig = superpose_volumes(receptor, rligand, t_correct)
	sup_rec[ sup_lig>=0.9 ] = 0
	
	return receptor, ligand, t_correct, rotation_angle_correct


def test_complex_generation():
	receptor, ligand, t_correct, rotation_angle_correct = generate_complex()

	rligand = rotate_ligand(ligand, rotation_angle_correct)
	
	dx, dy = int(t_correct[0]), int(t_correct[1])
	print('Translation=', dx, dy, 'Rotation:', rotation_angle_correct)
	
	field = np.zeros( (200, 200) )
	field[75:125, 75:125] = receptor
	field[  100 + dx - 25: 100 + dx + 25, 
			100 + dy - 25: 100 + dy + 25] += 2*rligand
	field[150:200,150:200]  = receptor
	field[150:200,100:150]  = rligand
	plt.imshow(field)
	plt.show()
		


if __name__=='__main__':
	test_superpos()
	#test_rotations()
	#test_translation_picking()
	#test_complex_generation()