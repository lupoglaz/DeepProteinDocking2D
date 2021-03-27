import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl
from pathlib import Path

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

import random
from random import uniform

from Protein import Protein
from Complex import Complex
from DockerGPU import DockerGPU
from Interaction import Interaction
from tqdm import tqdm

def generate_dataset(dataset_name, num_examples=100, min_gap=0.2, rmsd_cutoff=5.0, boundary_size=3, a00=1.0, a11=0.4, a10=-1.0):
	if Path(dataset_name).exists():
		with open(dataset_name, 'rb') as fin:
			dataset = pkl.load(fin)
		return dataset
		
	def generate_example():
		while(True):
			rec = Protein.generateConcave(size=50, num_points = 100, alpha=0.95)
			lig = Protein.generateConcave(size=50, num_points = 100, alpha=0.95)
			cplx = Complex.generate(rec, lig)

			dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
			scores = dck.dock_global(rec, lig)
			
			#RMSD filter
			best_score, dock_cplx, ind = dck.get_conformation(scores, rec, lig)
			rmsd = lig.rmsd(cplx.translation, cplx.rotation, dock_cplx.translation, dock_cplx.rotation)
			if rmsd > rmsd_cutoff:
				continue

			#Funnel gap filter
			inter = Interaction(dck, scores)
			funnels, complexes = inter.find_funnels(num_funnels=2)
			if len(funnels) < 2:
				continue
			min_score0 = torch.min(funnels[0][1]).item()
			min_score1 = torch.min(funnels[1][1]).item()
			if min_score1 - min_score0 < min_gap*np.abs(min_score0):
				continue
						
			return dock_cplx.receptor.bulk, dock_cplx.ligand.bulk, dock_cplx.translation, dock_cplx.rotation
	
	dataset = [generate_example() for i in tqdm(range(num_examples))]
	
	with open(dataset_name, 'wb') as fout:
		dataset = pkl.dump(dataset, fout)

	return dataset

def interaction_dataset(dataset_name):
	with open(dataset_name, 'rb') as fin:
		dataset = pkl.load(fin)

	all_proteins = []
	native_scores = []
	for receptor, ligand, rotation, translation in tqdm(dataset):
		all_proteins.append(Protein(receptor))
		all_proteins.append(Protein(ligand))
		dck = DockerGPU(Protein(receptor), Protein(ligand), num_angles=360, boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
		scores = dck.dock_global()
		score, cplx_docked, ind = dck.get_conformation(scores)
		native_scores.append(score)

	max_native_score = max(native_scores)

	
	all_proteins = all_proteins
	num_proteins = len(all_proteins)
	non_native_scores = []
	interaction_matrix = np.zeros((num_proteins,num_proteins))
	for rec_ind, receptor in tqdm(enumerate(all_proteins)):
		for lig_ind, ligand in enumerate(all_proteins):
			if lig_ind == (rec_ind + 1): #belongs to the native interaction
				interaction_matrix[rec_ind, lig_ind] = 1.0#native_scores[int(rec_ind/2)]
				continue
			if lig_ind > rec_ind: #avoid double counting
				continue
			
			dck = DockerGPU(receptor, ligand, num_angles=360, boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
			scores = dck.dock_global()
			score, cplx_docked, ind = dck.get_conformation(scores)
			non_native_scores.append(score)
			if score < max_native_score:
				interaction_matrix[rec_ind, lig_ind] = 1.0
				interaction_matrix[lig_ind, rec_ind] = 1.0

	plt.figure(figsize=(12,6))
	plt.subplot(1,2,1)
	plt.title(f'Scores')
	# plt.hist(non_native_scores+native_scores, label='all')
	plt.hist(native_scores, label='native')
	plt.hist(non_native_scores, label='non-native')
	plt.legend()

	plt.subplot(1,2,2)
	plt.title(f'Interaction')
	plt.imshow(interaction_matrix)
	plt.colorbar()
	plt.savefig('Results/interactions.png')

if __name__=='__main__':
	# generate_dataset('dock_validation.pkl', num_examples=100, min_gap=0.2, rmsd_cutoff=5.0)
	interaction_dataset('dock_validation.pkl')