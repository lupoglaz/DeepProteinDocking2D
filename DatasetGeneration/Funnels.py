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
from DockerGPU import DockerGPU
from tqdm import tqdm


class Interaction:
	def __init__(self, docker, scores):
		self.docker = docker
		self.scores = scores

		self.min_score, self.cplx, self.ind = self.docker.get_conformation(self.scores)
		
	@classmethod
	def with_docker(cls, docker):
		min_score, cplx_docked, scores, min_ind = docker.dock_global()
		return cls(docker, scores)
		
	def find_funnels(self, num_funnels=2):
		rmsd_all = self.docker.ligand.compute_rmsd(self.docker.angles, self.cplx.translation, torch.tensor([self.cplx.rotation])).to(device='cuda')

		funnels = []
		complexes = []
		funnel_scores = self.scores.clone()
				
		for i in range(num_funnels):
			funnel_min_score, cplx, ind = self.docker.get_conformation(funnel_scores)
			print(funnel_min_score, ind)
			funnel_trans = cplx.translation
			funnel_rot = torch.tensor([cplx.rotation])

			rmsd_grid = self.docker.ligand.compute_rmsd(self.docker.angles, funnel_trans, funnel_rot).to(device='cuda')

			mask_scores_clus = funnel_scores < 0.9*funnel_min_score
			mask_rmsd = rmsd_grid < 5.0
			mask_funnel = torch.logical_and(mask_rmsd, mask_scores_clus)
			
			funnel_rmsd = rmsd_all.masked_select(mask_funnel).clone()
			funnel_sc = funnel_scores.masked_select(mask_funnel).clone()
			
			funnel_scores = funnel_scores.masked_fill(mask_rmsd, 0.0)
			
			complexes.append(cplx)
			funnels.append((funnel_rmsd, funnel_sc))
		
		return funnels, complexes

def plot_funnels(docker, scores, ind, cplx):
	min_score = scores[ind[0], ind[1], ind[2]]
	mask_scores = scores < 0.3*min_score
	rmsd_grid = docker.ligand.compute_rmsd(docker.angles, cplx.translation, torch.tensor([cplx.rotation])).to(device='cuda')
	
	all_rmsd = rmsd_grid.masked_select(mask_scores)
	all_sc = scores.masked_select(mask_scores)
	
	inter = Interaction(docker, scores)
	funnels, complexes = inter.find_funnels()
	
	for i,cplx in enumerate(complexes):
		plt.subplot(1,3,i+1)
		plt.title(f'Complex #{i}')
		cplx.plot()
	
	plt.subplot(1,3,3)
	plt.scatter(all_rmsd.cpu().numpy(), all_sc.cpu().numpy())
	for i, funnel in enumerate(funnels):
		plt.scatter(funnel[0].cpu().numpy(), funnel[1].cpu().numpy(), label=f'Funnel:{i}')
	plt.legend()

def test_dock_global():
	rec = Protein.generateConcave(size=50, alpha=0.90, num_points = 100)
	lig = Protein.generateConcave(size=50, alpha=0.90, num_points = 100)
	cplx = Complex.generate(rec, lig)
	cor_score = cplx.score(boundary_size=3, weight_bulk=4.0)
	
	dck = DockerGPU(cplx.receptor, cplx.ligand, num_angles=360, boundary_size=3, weight_bulk=4.0)
	score, cplx_docked, scores, ind = dck.dock_global()
	docked_score = cplx_docked.score(boundary_size=3, weight_bulk=4.0)

	print('Predicted:')
	print(f'Score:{score}/{docked_score}', 'Translation:', cplx_docked.translation, 'Rotation:', cplx_docked.rotation)
	print('Correct:')
	print('Score:', cor_score, 'Translation:', cplx.translation, 'Rotation:', cplx.rotation)
	plt.figure(figsize=(12,6))
	# plt.subplot(1,3,1)
	# plt.title(f'{cor_score}')
	# cplx.plot()
	
	# plt.subplot(1,3,2)
	# plt.title(f'{score}')
	# cplx_docked.plot()
	
	# plt.subplot(1,3,3)
	# plt.title(f'Funnels')
	plot_funnels(dck, scores, ind, cplx_docked)
	plt.show()
	

def dataset(dataset_name):
	with open(dataset_name, 'rb') as fin:
		dataset = pkl.load(fin)
	
	for receptor, ligand, rotation, translation in dataset:
		dck = DockerGPU(Protein(receptor), Protein(ligand), num_angles=360, boundary_size=3, weight_bulk=4.0)
		score, cplx_docked, scores, ind = dck.dock_global()
		docked_score = cplx_docked.score(boundary_size=3, weight_bulk=4.0)

		print('Predicted:')
		print(f'Score:{score}/{docked_score}', 'Translation:', cplx_docked.translation, 'Rotation:', cplx_docked.rotation)
		plt.figure(figsize=(12,6))
		plt.subplot(1,2,1)
		plt.title(f'{score}')
		cplx_docked.plot()
		
		plt.subplot(1,2,2)
		plt.title(f'Funnels')
		plot_funnels(dck, scores, ind, cplx_docked)
		plt.show()

		break

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


if __name__=='__main__':
	test_dock_global()
	# interaction_dataset('toy_dataset_valid.pkl')