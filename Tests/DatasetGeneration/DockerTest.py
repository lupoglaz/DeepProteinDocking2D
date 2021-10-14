import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl
import math

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from DatasetGeneration import DockerGPU, Protein, Complex

def test_dock_global(a00, a11, a10, boundary_size=3):
	rec = Protein.generateConcave(size=50, num_points = 100)
	lig = Protein.generateConcave(size=50, num_points = 100)
	cplx = Complex.generate(rec, lig)
	cor_score = cplx.score(boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)

	dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
	scores = dck.dock_global(cplx.receptor, cplx.ligand)
	score, cplx_docked, ind = dck.get_conformation(scores, cplx.receptor, cplx.ligand)
	docked_score = cplx_docked.score(boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)

	rmsd = lig.rmsd(cplx.translation, cplx.rotation, cplx_docked.translation, cplx_docked.rotation)

	print('Predicted:')
	print(f'Score:{score}/{docked_score}', 'Translation:', cplx_docked.translation, 'Rotation:', cplx_docked.rotation, 'RMSD:', rmsd)
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

def test_rotation(Nrot=10):
	"""
	Comparing DockerGPU rotation to Protein rotation
	"""
	ligand = Protein.generateConcave(size=50)
	field = np.zeros((100, 50+Nrot*50))
	field[:50,:50] = ligand.bulk
	
	angles = torch.zeros(Nrot, dtype=torch.float)
	for i in range(Nrot):
		rotation_angle = (i+1)*360.0/Nrot
		rligand = ligand.rotate(np.pi*rotation_angle/180.0)
		angles[i] = np.pi*rotation_angle/180.0
		field[:50,(i+1)*50:(i+2)*50] = rligand.bulk
	
	docker = DockerGPU()
	tligand = torch.from_numpy(ligand.bulk).unsqueeze(dim=0).unsqueeze(dim=0).repeat(Nrot, 1, 1, 1).to(dtype=torch.float)
	dligand = docker.rotate(tligand, angles)
	field[50:,:50] = tligand[0,0,:,:]
	for i in range(Nrot):
		field[50:,(i+1)*50:(i+2)*50] = dligand[i,0,:,:]

	plt.imshow(field)
	plt.show()

def test_data_docking(dataset, a00, a11, a10):
	cell_size = 90
	canvas = np.zeros((cell_size, 2*cell_size))

	rmsds = []
	for plot_num, (receptor, ligand, translation, rotation) in enumerate(dataset):
		rec = Protein(receptor)
		lig = Protein(ligand)
		cplx = Complex(rec, lig, rotation, translation)
		dck = DockerGPU(num_angles=360, a00=a00, a11=a11, a10=a10)
		scores = dck.dock_global(rec, lig)
		best_score, res_cplx, ind = dck.get_conformation(scores, rec, lig)
			
		canvas[:, :cell_size] = cplx.get_canvas(cell_size)
		canvas[:, cell_size:] = res_cplx.get_canvas(cell_size)

		plt.imshow(canvas)	
		plt.show()

def test_data_FI(dataset, a00, a11, a10):
	prots, mat = dataset
	print(mat.shape)
	positive = []
	negative = []
	N_mat = 25
	N_pair = 20
	for i in range(N_mat):
		for j in range(N_mat):
			if mat[i,j] == 0:
				negative.append((i,j))
			else:
				positive.append((i,j))
	negative = negative[:N_pair]
	positive = positive[:N_pair]

	res_pos = []
	res_neg = []
	for i,j in positive + negative:
		rec = Protein(prots[i])
		lig = Protein(prots[j])
		dck = DockerGPU(num_angles=360, a00=a00, a11=a11, a10=a10)
		scores = dck.dock_global(rec, lig)
		best_score, res_cplx, ind = dck.get_conformation(scores, rec, lig)	
		if (i,j) in positive:
			res_pos.append(best_score)
		else:
			res_neg.append(best_score)
	
	plt.scatter(np.arange(N_pair), res_pos, label='pos')
	plt.scatter(np.arange(N_pair), res_neg, label='neg')
	plt.legend()
	plt.show()

if __name__=='__main__':
	# test_rotation(Nrot=10)
	# test_dock_global(a00=3.0, a10=1.1, a11=-0.3)
	# with open("../../DatasetGeneration/docking_data_test.pkl", 'rb') as fin:
	# 	dataset = pkl.load(fin)
	# 	test_data_docking(dataset, a00=3.0, a10=1.1, a11=-0.3)

	with open("../../DatasetGeneration/interaction_data_test.pkl", 'rb') as fin:
		dataset = pkl.load(fin)
		test_data_FI(dataset, a00=3.0, a10=1.1, a11=-0.3)
	