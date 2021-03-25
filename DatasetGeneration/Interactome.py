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
from Funnels import Interaction
from tqdm import tqdm


def interaction_dataset(dataset_name):
	with open(dataset_name, 'rb') as fin:
		dataset = pkl.load(fin)

	all_proteins = []
	native_scores = []
	for receptor, ligand, rotation, translation in tqdm(dataset):
		all_proteins.append(Protein(receptor))
		all_proteins.append(Protein(ligand))
		dck = DockerGPU(Protein(receptor), Protein(ligand), num_angles=360, boundary_size=3, weight_bulk=4.0)
		score, cplx_docked, scores, ind = dck.dock_global()
		native_scores.append(score)

	max_native_score = max(native_scores)

	
	all_proteins = all_proteins#[:20]
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
			
			dck = DockerGPU(receptor, ligand, num_angles=360, boundary_size=3, weight_bulk=4.0)
			score, cplx_docked, scores, ind = dck.dock_global()
			non_native_scores.append(score)
			if score < max_native_score:
				interaction_matrix[rec_ind, lig_ind] = 1.0
				interaction_matrix[lig_ind, rec_ind] = 1.0

	plt.figure(figsize=(12,6))
	plt.subplot(1,2,1)
	plt.title(f'Scores')
	plt.hist(non_native_scores+native_scores)
	plt.hist(native_scores)
	plt.hist(non_native_scores)

	plt.subplot(1,2,2)
	plt.title(f'Interaction')
	plt.imshow(interaction_matrix)
	plt.colorbar()
	plt.show()