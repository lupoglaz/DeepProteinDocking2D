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
from Protein import Protein
from Complex import Complex
from tqdm import tqdm

class Docker:
	def __init__(self, receptor, ligand,
				num_angles=10, weight_bulk=1.0, boundary_size=1):
		
		self.num_angles = num_angles
		self.weight_bulk = weight_bulk
		self.receptor = receptor
		self.ligand = ligand
		self.receptor.make_boundary(boundary_size=boundary_size)
		self.ligand.make_boundary(boundary_size=boundary_size)
				
		self.angles = np.linspace(-np.pi, np.pi, num=num_angles)
		self.Z = 0.0
	
	def dock_global(self):
		scores = []
		translations = []
		f_rec = self.receptor.get_repr()
		for i in range(self.num_angles):
			rlig = self.ligand.rotate(self.angles[i])
			rot_f_lig = rlig.get_repr()
			translation, score = self.dock_translations(f_rec, rot_f_lig)
			scores.append(score)
			translations.append(translation)

		min_ind = np.argmin(scores)
		best_translation = translations[min_ind]
		best_rotation = self.angles[min_ind]
		best_score = scores[min_ind]
		res_cplx = Complex(self.receptor, self.ligand, best_rotation, best_translation)
		
		return best_score, res_cplx

	def dock_translations(self, receptor, ligand):
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
		cplx_rec = np.fft.rfft2(receptor[1,:,:])
		cplx_lig = np.fft.rfft2(ligand[0,:,:])
		trans_bound_bulk = np.fft.irfft2(cplx_rec * np.conj(cplx_lig))
		
		score = -trans_bound[:,:] - 0.5*(trans_bulk_bound + trans_bound_bulk) + self.weight_bulk*trans_bulk
		ind = np.unravel_index(np.argmin(score, axis=None), score.shape)
		
		translation = np.array([ind[0], ind[1]])
		if translation[0]>=box_size:
			translation[0] -= 2*box_size
		if translation[1]>=box_size:
			translation[1] -= 2*box_size
		
		return translation, score[ind]


def test_dock_global():
	rec = Protein.generateConcave(size=50, num_points = 100)
	lig = Protein.generateConcave(size=50, num_points = 100)
	cplx = Complex.generate(rec, lig)
	cor_score = cplx.score(boundary_size=3, weight_bulk=4.0)
	# cplx.plot()
	dck = Docker(cplx.receptor, cplx.ligand, num_angles=360, boundary_size=3, weight_bulk=4.0)
	score, cplx_docked = dck.dock_global()
	docked_score = cplx_docked.score(boundary_size=3, weight_bulk=4.0)

	print('Predicted:')
	print(f'Score:{score}/{docked_score}', 'Translation:', cplx_docked.translation, 'Rotation:', cplx_docked.rotation)
	print('Correct:')
	print('Score:', cor_score, 'Translation:', cplx.translation, 'Rotation:', cplx.rotation)
	plt.figure(figsize=(12,6))
	plt.subplot(1,2,1)
	plt.title(f'{cor_score}')
	cplx.plot()
	plt.subplot(1,2,2)
	plt.title(f'{score}')
	cplx_docked.plot()
	plt.show()


def generate_dataset(num_examples=1000):
	def generate_example(filter=True):
		while(True):
			rec = Protein.generateConcave(size=50, num_points = 100)
			lig = Protein.generateConcave(size=50, num_points = 100)
			cplx = Complex.generate(rec, lig)
			dck = Docker(cplx.receptor, cplx.ligand, num_angles=360, boundary_size=3, weight_bulk=4.0)
			score, cplx_docked = dck.dock_global()
						
			if filter:
				if np.linalg.norm(cplx_docked.translation - cplx.translation)<5.0 and np.abs(cplx_docked.rotation-cplx.rotation)<(np.pi*10.0/180.0):
					return cplx_docked.receptor.bulk, cplx_docked.ligand.bulk, cplx_docked.translation, cplx_docked.rotation
			else:
				return cplx_docked.receptor.bulk, cplx_docked.ligand.bulk, cplx_docked.translation, cplx_docked.rotation
	
	dataset = [generate_example() for i in tqdm(range(num_examples))]
	return dataset

def reformat(dataset_name):
	with open(dataset_name, 'rb') as fin:
		dataset = pkl.load(fin)
	
	reformatted_data = []
	for receptor, ligand, translation, rotation in dataset:
		reformatted_data.append((receptor.bulk, ligand.bulk, translation, rotation))

	with open(dataset_name, 'wb') as fout:
		pkl.dump(reformatted_data, fout)

def generate(dataset_name, num_examples):
	dataset = generate_dataset(num_examples=num_examples)
	reformatted_data = []
	for receptor, ligand, translation, rotation in dataset:
		reformatted_data.append((receptor.bulk, ligand.bulk, translation, rotation))

	with open(dataset_name, 'wb') as fout:
		pkl.dump(reformatted_data, fout)

if __name__=='__main__':
	
	generate('toy_dataset_train.pkl', 1000)
	generate('toy_dataset_valid.pkl', 100)
	# test_dock_global()
	# reformat('unfilt_toy_dataset_valid.pkl')
	# reformat('unfilt_toy_dataset_1000.pkl')
	