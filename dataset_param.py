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

def plot_param_scan(input_name, ax, output_name=None, name="", ylabel=True):
	import matplotlib.font_manager as font_manager
	from matplotlib import rcParams
	font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
	rcParams['font.family'] = 'Helvetica'
	font = {'weight': 'normal',
			'size': 6}
	with open(input_name, 'rb') as fin:
		a00, a10, a11, M = pkl.load(fin)
	
	extent=(a10[0], a10[-1], a11[0], a11[-1])
	p = ax.imshow(M, extent=extent, origin='lower')
	ax.set_title(name)
	ax.set_xlabel('boundary-bulk')
	if ylabel:
		ax.set_ylabel('boundary-boundary')
	ax.set_xticks(a10)
	ax.set_yticks(a11)
	ax.tick_params(axis='x', which='major', labelsize=8, rotation=90)
	ax.tick_params(axis='x', which='minor', labelsize=8, rotation=90)
	ax.tick_params(axis='y', which='major', labelsize=8)
	ax.tick_params(axis='y', which='minor', labelsize=8)
	return p

def param_scans(input_names, titles=["", ""], output_name=None):
	import matplotlib.font_manager as font_manager
	from matplotlib import rcParams
	font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
	rcParams['font.family'] = 'Helvetica'
	fig, axs = plt.subplots(1, 2, figsize=(8,8), sharey=True, constrained_layout=True)
	pa = plot_param_scan(input_names[0], ax=axs[0], name=titles[0])
	pb = plot_param_scan(input_names[1], ax=axs[1], name=titles[1], ylabel=False)
	fig.colorbar(pa, ax=axs[0], shrink=0.6, location='bottom')
	fig.colorbar(pb, ax=axs[1], shrink=0.6, location='bottom')
	# plt.tight_layout()
	if output_name is None:
		plt.show()
	else:
		plt.savefig(output_name)


if __name__=='__main__':

	# cplx_scan_param(param=np.arange(0.2, 0.6, 0.05, dtype=np.float32),
	# 				func=lambda x, y, p: Complex.generate(x, y, threshold=p),
	# 				num_samples=10,
	# 				output_name='comp_overlap.png', name='Overlap fraction')

	# dataset = generate_dataset('DatasetGeneration/Data/score_param_prots.pkl', num_examples=100, overlap=0.4)
	
	# scan_parameters(dataset, get_rmsd, output_name='DatasetGeneration/Data/score_param_rmsd.pkl',
	# 				a11=np.arange(-3.0, 3.0, 0.1), a10=np.arange(-3.0, 0.0, 0.1), a00=3.0, boundary_size=3, num_samples=20)
	
	plot_dock_examples(dataset, a00=3.0, a10=-0.3, a11=2.5)
	
	
	# scan_parameters(dataset, get_funnel_gap, output_name='DatasetGeneration/Data/score_param_gap.pkl',
	# 				a11=np.arange(-3.0, 3.0, 0.1), a10=np.arange(-3.0, 0.0, 0.1), a00=3.0, boundary_size=3, num_samples=20)

	# plot_funnel_examples(dataset, a00=3.0, a10=-0.3, a11=2.5)

	# plot_param_scan(input_name='DatasetGeneration/Data/score_a00=10.0_param_rmsd.pkl',
	# 				output_name='score_param_rmsd_a10.png', name='RMSD, a00=10.0')
	# plot_param_scan(input_name='DatasetGeneration/Data/score_a00=10.0_param_gap.pkl',
	# 				output_name='score_param_gap_a10.png', name='Funnel gap, a00=10.0')
	# param_scans(input_names=['DatasetGeneration/Data/score_a00=3.0_param_rmsd.pkl', 'DatasetGeneration/Data/score_a00=3.0_param_gap.pkl'],
	# 			titles=['RMSD', 'Funnel gap'], output_name='dataset_param_scan.png')