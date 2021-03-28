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

from pathlib import Path


class Interactome:
	def __init__(self, docker, proteins=[]):
		self.proteins = proteins
		self.docker = docker
		
	@classmethod
	def from_dataset(cls, docker, filename):
		with open(filename, 'rb') as fin:
			dataset = pkl.load(fin)
		
		proteins = []
		for receptor, ligand, rotation, translation in dataset:
			proteins.append(receptor)
			proteins.append(ligand)
			
		return cls(docker, proteins)
	
	def non_native(self):
		for rec_ind, receptor in tqdm(enumerate(self.proteins), total=len(self.proteins)):
			for lig_ind, ligand in tqdm(enumerate(self.proteins), total=rec_ind, leave=False):
				if lig_ind > rec_ind: #avoid double counting
					continue
				if (lig_ind + 1 == rec_ind) and (rec_ind%2 == 1): #belongs to native 
					continue
				receptor, ligand = Protein(receptor), Protein(ligand)
				scores = self.docker.dock_global(receptor, ligand)
				yield rec_ind, lig_ind, Interaction(self.docker, scores, receptor, ligand)
	
	def native(self):
		for n in tqdm(range(0, len(self.proteins), 2)):
			receptor, ligand = Protein(self.proteins[n]), Protein(self.proteins[n+1])
			scores = self.docker.dock_global(receptor, ligand)
			yield n, n+1, Interaction(self.docker, scores, receptor, ligand)

	def all(self):
		for rec_ind, receptor in tqdm(enumerate(self.proteins), total=len(self.proteins)):
			for lig_ind, ligand in tqdm(enumerate(self.proteins), total=rec_ind, leave=False):
				if lig_ind > rec_ind: #avoid double counting
					continue
				rec, lig = Protein(receptor), Protein(ligand)
				scores = self.docker.dock_global(rec, lig)
				yield rec_ind, lig_ind, Interaction(self.docker, scores, rec, lig)
			

if __name__=='__main__':
	docker = DockerGPU(boundary_size=3, a00=1.0, a11=0.4, a10=-1.0)
	interactome = Interactome.from_dataset(docker, Path('dock_validation.pkl'))
	for i, j, interaction in interactome.native():
		plt.figure(figsize=(12,6))
		interaction.plot_funnels(num_funnels=3)
		print(interaction.est_binding(5.0))
		plt.show()
		# break