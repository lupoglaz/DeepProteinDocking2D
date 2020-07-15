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
from complex_generator import generate_complex, rotate_ligand
from prot_generator import generate_protein

from tqdm import tqdm

def protein_representation(protein, boundary_size=3):
	size = protein.shape[0]
	boundary = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			if protein[i,j]>0.5:
				continue
			x_low = max(i - boundary_size, 0)
			x_high = min(i + boundary_size, size)
			y_low = max(j - boundary_size, 0)
			y_high = min(j + boundary_size, size)
			if np.sum(protein[x_low:x_high, y_low:y_high])>0.5:
				boundary[i,j]=1.0
	features = np.stack([protein, boundary], axis=0)
	return features

def test_representation():
	receptor = generate_protein(size=50)
	features = protein_representation(receptor)
	
	field = np.zeros( (50, 100) )
	field[:, 0:50] = features[0,:,:]
	field[:, 50:100] = features[1,:,:]
	plt.imshow(field)
	plt.show()

def dock_translations(receptor, ligand, weight_bulk=-1.0):
	box_size = receptor.shape[-1]
	receptor = np.pad(receptor, ((0,0), (0, box_size), (0, box_size)), 'constant', constant_values=((0, 0),(0, 0), (0,0)))
	ligand = np.pad(ligand, ((0,0), (0, box_size), (0, box_size)), 'constant', constant_values=((0, 0),(0, 0), (0,0)))
	
	#Bulk score
	cplx_rec = np.fft.rfft2(receptor[0,:,:])
	cplx_lig = np.fft.rfft2(ligand[0,:,:])
	trans_bulk = np.fft.irfft2(cplx_rec * np.conj(cplx_lig))

	#Boundary score
	cplx_rec = np.fft.rfft2(receptor[1,:,:])
	cplx_lig = np.fft.rfft2(ligand[1,:,:])
	trans_bound = np.fft.irfft2(cplx_rec * np.conj(cplx_lig))

	#Boundary - bulk score
	cplx_rec = np.fft.rfft2(receptor[0,:,:])
	cplx_lig = np.fft.rfft2(ligand[1,:,:])
	trans_bulk_bound = np.fft.irfft2(cplx_rec * np.conj(cplx_lig))

	#Bulk - boundary score
	cplx_rec = np.fft.rfft2(receptor[0,:,:])
	cplx_lig = np.fft.rfft2(ligand[1,:,:])
	trans_bound_bulk = np.fft.irfft2(cplx_rec * np.conj(cplx_lig))
	

	score = 0.5*trans_bound[:,:] + 0.5*(trans_bulk_bound + trans_bound_bulk) + weight_bulk*trans_bulk
	ind = np.unravel_index(np.argmax(score, axis=None), score.shape)
	
	translation = np.array([ind[0], ind[1]])
	if translation[0]>=box_size:
		translation[0] -= 2*box_size
	if translation[1]>=box_size:
		translation[1] -= 2*box_size
	
	return translation, score[ind]

def test_dock_translations():
	receptor, ligand, t_correct, rotation_angle_correct = generate_complex()
	rligand = rotate_ligand(ligand, rotation_angle_correct)
	f_rec = protein_representation(receptor, boundary_size=2)
	f_lig = protein_representation(rligand, boundary_size=2)
	translation, score = dock_translations(f_rec, f_lig)

	dx, dy = int(t_correct[0]), int(t_correct[1])
	print('Correct translation=', dx, dy)
	field = np.zeros( (200, 400) )
	field[75:125, 75:125] = receptor

	field[  100 + dx - 25: 100 + dx + 25, 
			100 + dy - 25: 100 + dy + 25] += 2*rligand
	
	dx, dy = int(translation[0]), int(translation[1])
	print('Predicted translation=', dx, dy)
	field[75:125, 275:325] = receptor
	field[  100 + dx - 25: 100 + dx + 25, 
			300 + dy - 25: 300 + dy + 25] += 4*rligand
	

	plt.imshow(field)
	plt.show()

def dock_global(receptor, ligand, num_angles=10, weight_bulk=-0.5, boundary_size=2):
	f_rec = protein_representation(receptor, boundary_size=boundary_size)
	f_lig = protein_representation(ligand, boundary_size=boundary_size)
	angles = np.linspace(0.0, 360.0, num=num_angles)

	scores = []
	translations = []
	for i in range(angles.shape[0]):
		rot_f_lig = np.stack([rotate_ligand(f_lig[j], angles[i]) for j in range(2)], axis=0)
		translation, score = dock_translations(f_rec, rot_f_lig)
		scores.append(score)
		translations.append(translation)

	max_ind = np.argmax(scores)
	best_translation = translations[max_ind]
	best_rotation = angles[max_ind]
	best_score = scores[max_ind]
	return best_score, best_translation, best_rotation, scores
	
def test_dock_global():
	receptor, ligand, t_correct, rotation_angle_correct = generate_complex()
	score, translation, rotation, all_scores = dock_global(receptor, ligand, num_angles=360, boundary_size=1)
	print('Predicted:')
	print('Score:', score, 'Translation:', translation, 'Rotation:', rotation)
	print('Correct:')
	print('Translation:', t_correct, 'Rotation:', rotation_angle_correct)

	rligand = rotate_ligand(ligand, rotation_angle_correct)
	dx, dy = int(t_correct[0]), int(t_correct[1])
	field = np.zeros( (200, 400) )
	field[75:125, 75:125] = receptor

	field[  100 + dx - 25: 100 + dx + 25, 
			100 + dy - 25: 100 + dy + 25] += 2*rligand
	
	rligand = rotate_ligand(ligand, rotation)
	dx, dy = int(translation[0]), int(translation[1])
	print('Predicted translation=', dx, dy)
	field[75:125, 275:325] = receptor
	field[  100 + dx - 25: 100 + dx + 25, 
			300 + dy - 25: 300 + dy + 25] += 2*rligand
	
	plt.figure(figsize=(10,5))
	plt.subplot(1,2,1)
	plt.imshow(field)
	plt.subplot(1,2,2)
	plt.hist(all_scores, bins=75)
	plt.tight_layout()
	plt.show()

if __name__=='__main__':
	# test_representation()
	# test_dock_translations()
	test_dock_global()
	