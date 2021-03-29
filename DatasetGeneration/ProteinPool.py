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
from Interactome import Interactome
from tqdm import tqdm
import inspect

class ParamDistribution:
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		
		all_attr = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		vars = [a for a in all_attr if not(a[0].startswith('__') and a[0].endswith('__'))]
		for param_name, distr in vars:
			self.normalize(param_name)
			
	def normalize(self, param_name):
		Z = 0.0
		param = getattr(self, param_name)
		for val, prob in param:
			Z += prob
		new_param = []
		for val, prob in param:
			new_param.append((val, prob/Z))
		setattr(self, param_name, new_param)

	def sample(self, param_name):
		param = getattr(self, param_name)
		vals, prob = zip(*param)
		return random.choices(vals, cum_weights=prob, k=1)[0]


class InteractionCriteria:
	def __init__(self, score_cutoff=-70, funnel_gap_cutoff=10):
		self.score_cutoff = score_cutoff
		self.funnel_gap_cutoff = funnel_gap_cutoff
	
	def __call__(self, interaction):
		G, score, gap = interaction
		if score < self.score_cutoff and gap>self.funnel_gap_cutoff:
			return True
		else:
			return False



class ProteinPool:
	def __init__(self, proteins):
		self.proteins = proteins
		self.params = []
		self.interactions = {}

	@classmethod
	def generate(cls, num_proteins, params, size=50):
		pool = cls([])
		for i in tqdm(range(num_proteins)):
			num_points = params.sample('num_points')
			alpha = params.sample('alpha')
			prot = Protein.generateConcave(size=size, num_points=num_points, alpha=alpha)
			pool.proteins.append(prot.bulk)
			pool.params.append({'alpha': alpha, 'num_points': num_points})
		return pool
	
	@classmethod
	def load(cls, filename):
		with open(filename, 'rb') as fin:
			proteins, params, interactions = pkl.load(fin)
		instance = cls(proteins)
		instance.params = params
		instance.interactions = interactions
		return instance
	
	def save(self, filename):
		with open(filename, 'wb') as fout:
			pkl.dump( (self.proteins, self.params, self.interactions), fout)

	def get_interactions(self, docker):
		interactome = Interactome(docker, proteins = self.proteins)
		for i, j, interaction in interactome.all():
			funnels, complexes = interaction.find_funnels(num_funnels=2)
			if len(funnels) == 2:
				funnel_gap = torch.min(funnels[1][1]).item() - torch.min(funnels[0][1]).item()
			else:
				funnel_gap = 0.0
			self.interactions[(i,j)] = (interaction.est_binding(2.0), interaction.min_score, funnel_gap)

	def create_complexes(self, params):
		new_proteins = []
		new_params = []
		for i in tqdm(range(0, len(self.proteins), 2)):
			rec = Protein(self.proteins[i])
			lig = Protein(self.proteins[i+1])
			overlap = params.sample('overlap')
			cplx = Complex.generate(rec, lig, threshold=overlap)
			new_proteins.append(cplx.receptor.bulk)
			new_proteins.append(cplx.ligand.bulk)
			new_params.append({'overlap':overlap}.update(self.params[i]))
			new_params.append({'translation': cplx.translation, 'rotation': cplx.rotation}.update(self.params[i+1]))
		new_pool = ProteinPool(new_proteins)
		new_pool.params = new_params
		return new_pool
	
	def extract_docking_dataset(self, docker, interaction_criteria, max_num_samples=1000):
		pairs = [key for key, inter in self.interactions.items() if interaction_criteria(inter)]
		dataset = []
		for i, j in tqdm(pairs, total=max_num_samples):
			receptor, ligand = Protein(self.proteins[i]), Protein(self.proteins[j])
			scores = docker.dock_global(receptor, ligand)
			min_score, cplx, ind = docker.get_conformation(scores, receptor, ligand)
			dataset.append((cplx.receptor.bulk.copy(), cplx.ligand.bulk.copy(), cplx.translation.copy(), cplx.rotation))
			if len(dataset) == max_num_samples:
				break
		return dataset

	def extract_interactome_dataset(self, interaction_criteria, ind_range=(0, 900)):
		proteins_sel = [protein for n, protein in enumerate(self.proteins) if n>=ind_range[0]]
		
		N = ind_range[1] - ind_range[0]
		mat = np.zeros((N,N))
		for (i,j), inter in self.interactions.items():
			if not(i >= ind_range[0] and i<ind_range[1]):
				continue
			if not(j >= ind_range[0] and j<ind_range[1]):
				continue
			if interaction_criteria(inter):
				mat[i-ind_range[0], j-ind_range[0]] = 1.0
				mat[j-ind_range[0], i-ind_range[0]] = 1.0
		
		return proteins_sel.copy(), mat.copy()

	def plot_interaction_dist(self, perc=0.8):
		assert not(self.interactions is None)
			
		dG, scores, gaps = zip(*[v for k, v in self.interactions.items()])

		GMat, SMat, FMat = tuple([np.zeros((len(self.proteins), len(self.proteins))) for i in range(3)])
		for (i,j), (dg, score, gap) in self.interactions.items():
			GMat[i,j], SMat[i,j], FMat[i,j] = dg, score, gap
			GMat[j,i], SMat[j,i], FMat[j,i] = dg, score, gap
		
		Gdist, Sdist, Fdist = [], [], []
		print(np.percentile(dG, perc), np.percentile(scores, 100-perc), np.percentile(gaps, perc))
		GInt = GMat>np.percentile(dG, perc)
		SInt = SMat<np.percentile(scores, 100-perc)
		FInt = FMat>np.percentile(gaps, perc)
		for i in range(len(self.proteins)):
			Gdist.append(np.sum(GInt[i,:]))
			Sdist.append(np.sum(SInt[i,:]))
			Fdist.append(np.sum(FInt[i,:]))
		
		plt.figure(figsize=(12,12))
		plt.subplot(3,3,1)
		plt.hist(dG, label='Est. binding energy')
		plt.ylabel('Binding energy')
		plt.legend()
		plt.subplot(3,3,2)
		plt.imshow(1-GInt)
		plt.subplot(3,3,3)
		plt.hist(Gdist, bins=50)

		plt.subplot(3,3,4)
		plt.hist(scores, label='Min score')
		plt.ylabel('Minimum score')
		plt.legend()
		plt.subplot(3,3,5)
		plt.imshow(1-SInt)
		plt.subplot(3,3,6)
		plt.hist(Sdist, bins=50)

		plt.subplot(3,3,7)
		plt.hist(gaps, label='Funnel gap')
		plt.ylabel('Funnel gap')
		plt.legend()
		plt.subplot(3,3,8)
		plt.imshow(1-FInt)
		plt.subplot(3,3,9)
		plt.hist(Fdist, bins=50)
		
		plt.tight_layout()
		plt.show()

	def plot_interactions(self, docker, num_plots=10, type='best'):
		plt.figure(figsize=(num_plots*3, 3))
		cell_size = 90
		canvas = np.zeros((cell_size, cell_size*num_plots))
		if type=='best':
			sorted_inter = sorted(list(self.interactions.items()), key=lambda x: x[1][1])
		elif type=='worst':
			sorted_inter = sorted(list(self.interactions.items()), key=lambda x: -x[1][1])
		else:
			raise Exception(f'Unknown type {type}')
		
		plot_num = 0
		min_scores = []
		for (i, j), (G, score, gap) in sorted_inter[:num_plots]:
			rec, lig = Protein(self.proteins[i]), Protein(self.proteins[j])
			scores = docker.dock_global(rec, lig)
			min_score, cplx, ind = docker.get_conformation(scores, rec, lig)
			canvas[:,plot_num*cell_size:(plot_num+1)*cell_size] = cplx.get_canvas(cell_size)
			min_scores.append(min_score)
			plot_num += 1
		
		plt.imshow(canvas)
		plt.xticks(ticks=[i*cell_size + cell_size/2 for i in range(num_plots)], labels=['%.1f'%s for s in min_scores])
		plt.xlabel('score')
		plt.show()

	def plot_sample_funnels(self, docker, filename='funnels.png', range=[(-100, -80), (-70, -40), (-32, -20)], titles=['A','B','C']):
		# plt.figure(figsize=(18,6))
		font = {'family': 'serif',
				'weight': 'normal',
				'size': 18,
				}

		fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18,6))
		fig.subplots_adjust(wspace=0)
		num_samples = len(range)
		sample_num = 0
		for (i, j), (G, score, gap) in self.interactions.items():
			if not(score>range[sample_num][0] and score<=range[sample_num][1]):
				continue
			rec, lig = Protein(self.proteins[i]), Protein(self.proteins[j])
			scores = docker.dock_global(rec, lig)
			inter = Interaction(docker, scores, rec, lig)
			
			if sample_num ==0:
				inter.plot_funnels(ax=axs[sample_num], im_offset=(70,25))
			else:
				inter.plot_funnels(ax=axs[sample_num], im_offset=(60, -75))
			
			axs[sample_num].set_title(titles[sample_num], fontdict=font)
			axs[sample_num].set_xlabel('RMSD', fontdict=font)
			# axs[sample_num].legend(prop=font)
			for label in axs[sample_num].get_xticklabels():
				label.set_fontproperties(font)

			for label in axs[sample_num].get_yticklabels():
				label.set_fontproperties(font)

			sample_num += 1
			if sample_num == num_samples:
				break

		axs[0].set_ylabel('Energy', fontdict=font)
		plt.tight_layout()
		plt.savefig(filename)



if __name__=='__main__':
	params = ParamDistribution(
		alpha = [(0.8, 1), (0.9, 9), (0.95, 4)],
		num_points = [(20, 1), (30, 2), (50, 4), (80, 8), (100, 6)],
		overlap = [(0.10, 2), (0.20, 3), (0.30, 4), (0.50, 2), (0.60, 1)]
		)
	
	docker = DockerGPU(boundary_size=3, a00=10.0, a11=0.4, a10=-1.0)
	#Generation
	# pool = ProteinPool.generate(num_proteins=1000, params=params)
	# pool.get_interactions(docker)
	# pool.save('protein_pool_huge.pkl')
	
	#Docking dataset
	# pool = ProteinPool.load('Data/protein_pool_huge.pkl')
	# inter = InteractionCriteria(score_cutoff=-70, funnel_gap_cutoff=10)
	# dataset_all = pool.extract_docking_dataset(docker, inter, max_num_samples=1100)
	# print(f'Total data length {len(dataset_all)}')
	# random.shuffle(dataset_all)
	# with open('docking_data_train.pkl', 'wb') as fout:
	# 	pkl.dump(dataset_all[:1000], fout)
	# with open('docking_data_valid.pkl', 'wb') as fout:
	# 	pkl.dump(dataset_all[1000:], fout)

	#Interaction dataset
	# pool = ProteinPool.load('Data/protein_pool_huge.pkl')
	# inter = InteractionCriteria(score_cutoff=-70, funnel_gap_cutoff=10)
	# interactome_train = pool.extract_interactome_dataset(inter, ind_range=(0, 900))
	# interactome_valid = pool.extract_interactome_dataset(inter, ind_range=(900, 1000))
	# with open('interaction_data_train.pkl', 'wb') as fout:
	# 	pkl.dump(interactome_train, fout)
	# with open('interaction_data_valid.pkl', 'wb') as fout:
	# 	pkl.dump(interactome_valid, fout)

	#Visualization
	# pool = ProteinPool.load('protein_pool_huge.pkl')
	# pool.plot_interaction_dist(perc=90)
	# pool.plot_interactions(docker, num_plots=20, type='best')
	# pool.plot_interactions(docker, num_plots=20, type='worst')
	# pool.plot_sample_funnels(docker, filename='funnels.png', range=[(-100, -80), (-70, -40), (-32, -20)])
	
