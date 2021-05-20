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

from .Protein import Protein
from .Complex import Complex
from .DockerGPU import DockerGPU
from .Interaction import Interaction
from tqdm import tqdm


def get_funnel_gap(datapoint, a00, a11, a10, boundary_size=3):
	rec = Protein(datapoint[0])
	lig = Protein(datapoint[1])
	dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
	inter = Interaction.with_docker(dck, rec, lig)
	funnels, complexes = inter.find_funnels(num_funnels=2)
	if len(funnels) == 2:
		min_score0 = torch.min(funnels[0][1]).item()
		min_score1 = torch.min(funnels[1][1]).item()
		return (min_score1 - min_score0)/min_score0
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


def scan_parameters(data, func, output_name='gap_score_param1.png', num_samples=10, boundary_size=3,
	a11=np.arange(-0.1, 0.1, 0.01), a10=np.arange(-0.1, -0.0, 0.01), a00=1.0):
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
	if a11 is None:
		a11 = np.arange(-0.1, 0.1, 0.01)
	if a10 is None:
		a10 = np.arange(-0.1, -0.0, 0.01)
	M = np.zeros((len(a11), len(a10)))
	print(M.shape)
	for xi, x in tqdm(enumerate(a11), total=len(a11)):
		for yi, y in tqdm(enumerate(a10), total=len(a10), leave=False):
			gaps = []
			samples = num_samples
			sel_data = random.choices(data, k=num_samples)
			for datapoint in sel_data:
				gap = func(datapoint, a00, x, y, boundary_size=boundary_size)
				if not(gap is None):
					gaps.append(gap)
				
			M[xi, yi] = np.mean(gaps)

	with open(output_name, 'wb') as fout:
		pkl.dump((a00, a10, a11, M), fout)


def generate_dataset(dataset_name, num_examples=100, overlap=0.3, num_points=100, alpha=0.95):
	if Path(dataset_name).exists():
		with open(dataset_name, 'rb') as fin:
			dataset = pkl.load(fin)
		return dataset
	def generate_example(filter=False):
		while(True):
			rec = Protein.generateConcave(size=50, num_points=num_points, alpha=alpha)
			lig = Protein.generateConcave(size=50, num_points=num_points, alpha=alpha)
			cplx = Complex.generate(rec, lig, threshold=overlap)
			return cplx.receptor.bulk, cplx.ligand.bulk, cplx.translation, cplx.rotation
	
	dataset = [generate_example() for i in tqdm(range(num_examples))]
	
	with open(dataset_name, 'wb') as fout:
		pkl.dump(dataset, fout)

	return dataset
	