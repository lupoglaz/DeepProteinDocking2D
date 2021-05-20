import sys
import os
import numpy as np
import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")
import _pickle as pkl 

from DatasetGeneration import test_dock_global, scan_parameters, generate_dataset, get_funnel_gap, get_rmsd
from DatasetGeneration.Interaction import test_funnels
from DatasetGeneration import Protein
from DatasetGeneration import DockerGPU
from DatasetGeneration import Complex
from DatasetGeneration.Complex import scan_parameter as cplx_scan_param
from DatasetGeneration.Interaction import Interaction

def plot_dock_examples(dataset, a00, a11, a10, boundary_size=3, num_plots=10, filename=None):
	plt.figure(figsize=(num_plots*3, 3))
	cell_size = 90
	canvas = np.zeros((2*cell_size, cell_size*num_plots))

	rmsds = []
	for plot_num, (receptor, ligand, translation, rotation) in enumerate(dataset):
		if plot_num == num_plots:
			break
		rec = Protein(receptor)
		lig = Protein(ligand)
		cplx = Complex(rec, lig, rotation, translation)
		dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
		scores = dck.dock_global(rec, lig)
		best_score, res_cplx, ind = dck.get_conformation(scores, rec, lig)
		rmsds.append(lig.rmsd(translation, rotation, res_cplx.translation, res_cplx.rotation))
		
		canvas[:cell_size, plot_num*cell_size:(plot_num+1)*cell_size] = res_cplx.get_canvas(cell_size)
		canvas[cell_size:2*cell_size, plot_num*cell_size:(plot_num+1)*cell_size] = cplx.get_canvas(cell_size)
		
	
	plt.imshow(canvas)
	plt.xticks(ticks=[i*cell_size + cell_size/2 for i in range(num_plots)], labels=['%.1f'%s for s in rmsds])
	plt.xlabel('RMSD')
	if filename is None:
		plt.show()
	else:
		plt.savefig(filename)

def plot_funnel_examples(dataset, a00, a11, a10, boundary_size=3, num_plots=10, filename=None):
	fig, axs = plt.subplots(1, num_plots, sharey=True, figsize=(num_plots*3,3))
	fig.subplots_adjust(wspace=0)
	font = {'family': 'serif',
			'weight': 'normal',
			'size': 18,
			}

	for plot_num, (receptor, ligand, translation, rotation) in enumerate(dataset):
		if plot_num == num_plots:
			break
		rec = Protein(receptor)
		lig = Protein(ligand)
		dck = DockerGPU(num_angles=360, boundary_size=boundary_size, a00=a00, a11=a11, a10=a10)
		inter = Interaction.with_docker(dck, rec, lig)
		inter.plot_funnels(ax=axs[plot_num], plot_conformations=False)
		
		# axs[plot_num].set_title(titles[plot_num], fontdict=font)
		axs[plot_num].set_xlabel('RMSD', fontdict=font)
		# axs[sample_num].legend(prop=font)
		for label in axs[plot_num].get_xticklabels():
			label.set_fontproperties(font)

		for label in axs[plot_num].get_yticklabels():
			label.set_fontproperties(font)

	axs[0].set_ylabel('Energy', fontdict=font)
	plt.tight_layout()
	if filename is None:
		plt.show()
	else:
		plt.savefig(filename)

def plot_param_scan(input_name, output_name=None, name=""):
	with open(input_name, 'rb') as fin:
		a00, a10, a11, M = pkl.load(fin)
	
	f = plt.figure(figsize=(12,12))
	extent=(a10[0], a10[-1], a11[0], a11[-1])
	plt.imshow(M, extent=extent, origin='lower')
	plt.colorbar()
	plt.title(name)
	plt.xlabel('bound-bulk')
	plt.ylabel('bound-bound')
	plt.xticks(a10, fontsize=5)
	plt.yticks(a11, fontsize=6)
	plt.tight_layout()
	if output_name is None:
		plt.show()
	else:
		plt.savefig(output_name)

if __name__=='__main__':

	# cplx_scan_param(param=np.arange(0.2, 0.6, 0.05, dtype=np.float32),
	# 				func=lambda x, y, p: Complex.generate(x, y, threshold=p),
	# 				num_samples=10,
	# 				output_name='comp_overlap.png', name='Overlap')

	dataset = generate_dataset('DatasetGeneration/Data/score_param_prots.pkl', num_examples=100, overlap=0.4)
	
	# scan_parameters(dataset, get_rmsd, output_name='DatasetGeneration/Data/score_param_rmsd.pkl',
	# 				a11=np.arange(-3.0, 3.0, 0.1), a10=np.arange(-3.0, 0.0, 0.1), a00=3.0, boundary_size=3, num_samples=20)
	
	plot_dock_examples(dataset, a00=3.0, a10=-0.2, a11=-0.8)
	
	
	# scan_parameters(dataset, get_funnel_gap, output_name='DatasetGeneration/Data/score_param_gap1.pkl',
	# 				a11=np.arange(-3.0, 3.0, 0.1), a10=np.arange(-3.0, 0.0, 0.1), a00=3.0, boundary_size=3, num_samples=20)

	# plot_funnel_examples(dataset, a00=3.0, a10=-0.2, a11=-0.8)

	# plot_param_scan(input_name='DatasetGeneration/Data/score_param_rmsd.pkl',
	# 				output_name='score_param_rmsd.png', name='RMSD, a00=3.0')
	# plot_param_scan(input_name='DatasetGeneration/Data/score_param_gap1.pkl',
	# 				output_name='score_param_gap.png', name='Funnel gap, a00=3.0')