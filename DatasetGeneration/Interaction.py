import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from random import uniform
from Protein import Protein
from Complex import Complex
from DockerGPU import DockerGPU
from tqdm import tqdm


class Interaction:
	def __init__(self, docker, scores, receptor, ligand):
		self.docker = docker
		self.scores = scores
		self.min_score, self.cplx, self.ind = self.docker.get_conformation(self.scores, receptor, ligand)
		
	@classmethod
	def with_docker(cls, docker, receptor, ligand):
		scores = docker.dock_global(receptor, ligand)
		return cls(docker, scores, receptor, ligand)
		
	def find_funnels(self, num_funnels=2):
		rmsd_all = self.cplx.ligand.grid_rmsd(self.docker.angles, self.cplx.translation, torch.tensor([self.cplx.rotation])).to(device='cuda')

		funnels = []
		complexes = []
		funnel_scores = self.scores.clone()
				
		for i in range(num_funnels):
			funnel_min_score, cplx, ind = self.docker.get_conformation(funnel_scores, self.cplx.receptor, self.cplx.ligand)
			funnel_trans = cplx.translation
			funnel_rot = torch.tensor([cplx.rotation])

			rmsd_grid = self.cplx.ligand.grid_rmsd(self.docker.angles, funnel_trans, funnel_rot).to(device='cuda')

			mask_scores_clus = funnel_scores < 0.9*funnel_min_score
			mask_rmsd = rmsd_grid < 8.0
			mask_funnel = torch.logical_and(mask_rmsd, mask_scores_clus)
			
			funnel_rmsd = rmsd_all.masked_select(mask_funnel).clone()
			funnel_sc = funnel_scores.masked_select(mask_funnel).clone()
			if funnel_rmsd.size(0) == 0 or funnel_rmsd.size(0) == 0:
				break
			funnel_scores = funnel_scores.masked_fill(mask_rmsd, 0.0)
			
			complexes.append(cplx)
			funnels.append((funnel_rmsd, funnel_sc))
		
		return funnels, complexes

	def est_binding(self, T):
		return torch.log(torch.sum(torch.exp(-(1.0/T)*self.scores))).item()

	def plot_funnels(self, num_funnels=2, cell_size=90, ax=None, im_offset=(70,25)):
		mask_scores = self.scores < -10
		rmsd_grid = self.cplx.ligand.grid_rmsd(self.docker.angles, self.cplx.translation, torch.tensor([self.cplx.rotation])).to(device='cuda')
	
		all_rmsd = rmsd_grid.masked_select(mask_scores)
		all_sc = self.scores.masked_select(mask_scores)
	
		funnels, complexes = self.find_funnels()
		
		if ax is None:
			ax = plt.subplot(111)

		ax.scatter(all_rmsd.cpu().numpy(), all_sc.cpu().numpy())
		for i, funnel in enumerate(funnels):
			cplx_img = complexes[i].get_canvas(cell_size)
			rmsds = funnel[0].cpu().numpy()
			scores = funnel[1].cpu().numpy()
			ax.scatter(rmsds, scores, label=f'Funnel:{i}')
			
			im = OffsetImage(cplx_img.copy(), zoom=1.0)
			# im.image.axes = ax
			ab = AnnotationBbox(im, (rmsds[0], scores[0]),
								xybox=im_offset,
								xycoords='data',
								boxcoords="offset points",
								pad=0.3,
								arrowprops=dict(arrowstyle="->",color='black',lw=2.5))
			ax.add_artist(ab)
		

def test_funnels():
	rec = Protein.generateConcave(size=50, alpha=0.95, num_points = 100)
	lig = Protein.generateConcave(size=50, alpha=0.95, num_points = 100)
	cplx = Complex.generate(rec, lig)
	cor_score = cplx.score(boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
	
	dck = DockerGPU(num_angles=360, boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
	scores = dck.dock_global(cplx.receptor, cplx.ligand)
	score, cplx_docked, ind = dck.get_conformation(scores, cplx.receptor, cplx.ligand)
	docked_score = cplx_docked.score(boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)

	print('Predicted:')
	print(f'Score:{score}/{docked_score}', 'Translation:', cplx_docked.translation, 'Rotation:', cplx_docked.rotation)
	print('Correct:')
	print('Score:', cor_score, 'Translation:', cplx.translation, 'Rotation:', cplx.rotation)
	
	plt.figure(figsize=(12,6))
	Interaction(dck, scores, cplx_docked.receptor, cplx_docked.ligand).plot_funnels()
	plt.show()

if __name__=='__main__':
	test_funnels()