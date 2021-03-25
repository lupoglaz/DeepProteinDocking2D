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
		rmsd_all = self.docker.ligand.grid_rmsd(self.docker.angles, self.cplx.translation, torch.tensor([self.cplx.rotation])).to(device='cuda')

		funnels = []
		complexes = []
		funnel_scores = self.scores.clone()
				
		for i in range(num_funnels):
			funnel_min_score, cplx, ind = self.docker.get_conformation(funnel_scores)
			funnel_trans = cplx.translation
			funnel_rot = torch.tensor([cplx.rotation])

			rmsd_grid = self.docker.ligand.grid_rmsd(self.docker.angles, funnel_trans, funnel_rot).to(device='cuda')

			mask_scores_clus = funnel_scores < 0.9*funnel_min_score
			mask_rmsd = rmsd_grid < 5.0
			mask_funnel = torch.logical_and(mask_rmsd, mask_scores_clus)
			
			funnel_rmsd = rmsd_all.masked_select(mask_funnel).clone()
			funnel_sc = funnel_scores.masked_select(mask_funnel).clone()
			if funnel_rmsd.size(0) == 0 or funnel_rmsd.size(0) == 0:
				break
			funnel_scores = funnel_scores.masked_fill(mask_rmsd, 0.0)
			
			complexes.append(cplx)
			funnels.append((funnel_rmsd, funnel_sc))
		
		return funnels, complexes

def plot_funnels(docker, scores, ind, cplx):
	min_score = scores[ind[0], ind[1], ind[2]]
	mask_scores = scores < 0.3*min_score
	rmsd_grid = docker.ligand.grid_rmsd(docker.angles, cplx.translation, torch.tensor([cplx.rotation])).to(device='cuda')
	
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

def test_funnels():
	rec = Protein.generateConcave(size=50, alpha=0.95, num_points = 100)
	lig = Protein.generateConcave(size=50, alpha=0.95, num_points = 100)
	cplx = Complex.generate(rec, lig)
	cor_score = cplx.score(boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
	
	dck = DockerGPU(cplx.receptor, cplx.ligand, num_angles=360, boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
	scores = dck.dock_global()
	score, cplx_docked, ind = dck.get_conformation(scores)
	docked_score = cplx_docked.score(boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)

	print('Predicted:')
	print(f'Score:{score}/{docked_score}', 'Translation:', cplx_docked.translation, 'Rotation:', cplx_docked.rotation)
	print('Correct:')
	print('Score:', cor_score, 'Translation:', cplx.translation, 'Rotation:', cplx.rotation)
	
	plt.figure(figsize=(12,6))
	plot_funnels(dck, scores, ind, cplx_docked)
	plt.show()

if __name__=='__main__':
	test_funnels()