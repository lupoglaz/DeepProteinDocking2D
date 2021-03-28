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


def get_funnel_gap(datapoint, a00, a11, a10, boundary_size=3):
	rec = Protein(datapoint[0])
	lig = Protein(datapoint[1])
	dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
	scores = dck.dock_global(rec, lig)
	inter = Interaction(dck, scores)
	funnels, complexes = inter.find_funnels(num_funnels=2)
	if len(funnels) == 2:
		min_score0 = torch.min(funnels[0][1]).item()
		min_score1 = torch.min(funnels[1][1]).item()
		return min_score1 - min_score0
	else:
		return None

def get_rmsd(datapoint, a00, a11, a10, boundary_size=3):
	receptor, ligand, translation, rotation = datapoint
	rec = Protein(receptor)
	lig = Protein(ligand)
	dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
	scores = dck.dock_global(rec, lig)
	best_score, res_cplx, ind = dck.get_conformation(scores, rec, lig)
	rmsd = lig.rmsd(translation, rotation, res_cplx.translation, res_cplx.rotation)
	return rmsd.item()


def scan_parameters(data, func, output_name='gap_score_param1.png', num_samples=10, boundary_size=3, name=""):
	"""
	Dependence of gap in funnels on the scoring function parameters
	a11 = x
	a00 = 1.0
	a10 = y
	x = boundary-boundary energy
	y = boundary-bulk energy
	"""
	# xs = np.arange(-1.0, 1.0, 0.1)
	# ys = np.arange(-1.0, -0.0, 0.1)
	xs = np.arange(-0.1, 0.1, 0.01)
	ys = np.arange(-0.1, -0.0, 0.01)
	M = np.zeros((len(xs), len(ys)))
	
	for xi, x in tqdm(enumerate(xs), total=len(xs)):
		for yi, y in tqdm(enumerate(ys), total=len(ys), leave=False):
			gaps = []
			samples = num_samples
			sel_data = random.choices(data, k=num_samples)
			for datapoint in sel_data:
				gap = func(datapoint, 1.0, x, y, boundary_size=boundary_size)
				if not(gap is None):
					gaps.append(gap)
				
			M[xi, yi] = np.mean(gaps)
	
	f = plt.figure(figsize=(6,10))
	extent=(ys[0], ys[-1], xs[0], xs[-1])
	plt.imshow(M, extent=extent)
	plt.colorbar()
	plt.title(name)
	plt.xlabel('bound-bulk')
	plt.ylabel('bound-bound')
	plt.xticks(ys, fontsize=6)
	plt.yticks(xs, fontsize=9)
	plt.tight_layout()
	plt.savefig(output_name)


def generate_dataset(dataset_name, num_examples=100):
	if Path(dataset_name).exists():
		with open(dataset_name, 'rb') as fin:
			dataset = pkl.load(fin)
		return dataset
	def generate_example(filter=False):
		while(True):
			rec = Protein.generateConcave(size=50, num_points = 100, alpha=0.95)
			lig = Protein.generateConcave(size=50, num_points = 100, alpha=0.95)
			cplx = Complex.generate(rec, lig)
			return cplx.receptor.bulk, cplx.ligand.bulk, cplx.translation, cplx.rotation
	
	dataset = [generate_example() for i in tqdm(range(num_examples))]
	
	with open(dataset_name, 'wb') as fout:
		dataset = pkl.dump(dataset, fout)

	return dataset

if __name__ == '__main__':
	dataset = generate_dataset('test_dataset.pkl')

	# scan_parameters(dataset, get_funnel_gap, output_name='gap_score_param1.png', name='Funnel gap')
	# scan_parameters(dataset, get_rmsd, output_name='rmsd_nat_param_tiny.png', num_samples=30, name='RMSD')
	