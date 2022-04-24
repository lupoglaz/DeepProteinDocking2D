import numpy as np
import torch
import _pickle as pkl

import matplotlib.pylab as plt
from matplotlib import rcParams

import seaborn as sea
sea.set_style("whitegrid")

from .Protein import Protein
from .Complex import Complex
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
		return np.random.choice(vals, p=prob)


class InteractionCriteriaGandGap:
	def __init__(self, funnel_gap_cutoff=10, free_energy_cutoff=20):
		self.funnel_gap_cutoff = funnel_gap_cutoff
		self.free_energy_cutoff = free_energy_cutoff
	
	def __call__(self, interaction):
		G, score, gap = interaction
		if (G < self.free_energy_cutoff) and (gap > self.funnel_gap_cutoff):
			return True
		else:
			return False


class InteractionCriteriaG:
	def __init__(self, free_energy_cutoff=20):
		self.free_energy_cutoff = free_energy_cutoff
	
	def __call__(self, interaction):
		G, score, gap = interaction
		if (G < self.free_energy_cutoff):
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
			self.interactions[(i,j)] = (interaction.est_binding(), interaction.min_score, funnel_gap)

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
		proteins_sel = [protein for n, protein in enumerate(self.proteins) if ((n>=ind_range[0]) and (n<ind_range[1]))]
		
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

	def plot_interaction_dist(self, perc=80):
		assert not(self.interactions is None)
			
		dG, scores, gaps = zip(*[v for k, v in self.interactions.items()])

		GMat, SMat, FMat = tuple([np.zeros((len(self.proteins), len(self.proteins))) for i in range(3)])
		for (i,j), (dg, score, gap) in self.interactions.items():
			GMat[i,j], SMat[i,j], FMat[i,j] = dg, score, gap
			GMat[j,i], SMat[j,i], FMat[j,i] = dg, score, gap
		
		Gdist, Sdist, Fdist = [], [], []
		print(np.percentile(dG, 100-perc), np.percentile(scores, 100-perc), np.percentile(gaps, perc))
		GInt = GMat<np.percentile(dG, 100-perc)
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

	def plot_interactions(self, docker, filename=None, num_plots=10):
		import matplotlib.font_manager as font_manager
		# font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
		font_manager.fontManager.addfont('Helvetica.ttf')
		rcParams['font.family'] = 'Helvetica'
		
		plt.figure(figsize=(num_plots*2, 5))
		cell_size = 90
		canvas_best = np.zeros((cell_size, cell_size*num_plots))
		canvas_worst = np.zeros((cell_size, cell_size*num_plots))
		sorted_inter = sorted(list(self.interactions.items()), key=lambda x: x[1][0])
		
		plot_num = 0
		min_scores = []
		for (i, j), (G, score, gap) in sorted_inter[:num_plots]:
			rec, lig = Protein(self.proteins[i]), Protein(self.proteins[j])
			scores = docker.dock_global(rec, lig)
			min_score, cplx, ind = docker.get_conformation(scores, rec, lig)
			canvas_best[:,plot_num*cell_size:(plot_num+1)*cell_size] = cplx.get_canvas(cell_size)
			min_scores.append(G)
			plot_num += 1

		plot_num = 0
		max_scores = []
		for (i, j), (G, score, gap) in (sorted_inter[-num_plots:]):
			rec, lig = Protein(self.proteins[i]), Protein(self.proteins[j])
			scores = docker.dock_global(rec, lig)
			max_score, cplx, ind = docker.get_conformation(scores, rec, lig)
			canvas_worst[:,plot_num*cell_size:(plot_num+1)*cell_size] = cplx.get_canvas(cell_size)
			max_scores.append(G)
			plot_num += 1
		
		ax = plt.subplot(2,1,1)
		ax.grid(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(axis=u'both', which=u'both',length=0)
		plt.title('Interacting', fontsize=24)
		plt.imshow(canvas_best, origin='lower', interpolation='nearest', resample=False, filternorm=False, cmap='gist_heat_r')
		plt.xticks(ticks=[i*cell_size + cell_size/2 for i in range(num_plots)], labels=['%.1f'%s for s in min_scores], fontsize=16)
		plt.yticks([])
		ax = plt.subplot(2,1,2)
		ax.grid(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(axis=u'both', which=u'both',length=0)
		plt.title('Non-interacting', fontsize=24)
		plt.imshow(canvas_worst, origin='lower', interpolation='nearest', resample=False, filternorm=False, cmap='gist_heat_r')
		plt.xticks(ticks=[i*cell_size + cell_size/2 for i in range(num_plots)], labels=['%.1f'%s for s in max_scores], fontsize=16)
		# plt.xlabel('score', fontsize=16)
		plt.yticks([])
		plt.tight_layout()
		if filename is None:
			plt.show()
		else:
			plt.savefig(filename)

	def plot_sample_funnels(self, docker, filename='funnels.png', range=[(-100, -80), (-70, -40), (-32, -20)], titles=['A','B','C']):
		import matplotlib.font_manager as font_manager
		# font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
		font_manager.fontManager.addfont('Helvetica.ttf')
		rcParams['font.family'] = 'Helvetica'
		font = {'weight': 'normal',
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
			
			axs[sample_num].set_title(titles[sample_num], fontsize=24)
			axs[sample_num].set_xlabel('RMSD', fontsize=20)
			# axs[sample_num].legend(prop=font)
			for label in axs[sample_num].get_xticklabels():
				label.set_fontproperties(font)

			for label in axs[sample_num].get_yticklabels():
				label.set_fontproperties(font)

			sample_num += 1
			if sample_num == num_samples:
				break
			

		axs[0].set_ylabel('Energy', fontsize=20)
		plt.tight_layout()
		if filename is None:
			plt.show()
		else:
			plt.savefig(filename)

	def plot_params(self, output_name='stats.png'):
		from mpl_toolkits.axes_grid1 import make_axes_locatable
		import matplotlib.font_manager as font_manager
		# font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
		font_manager.fontManager.addfont('Helvetica.ttf')
		rcParams['font.family'] = 'Helvetica'
		alpha_lst = []
		num_pts_lst = []
		for param in self.params:
			alpha_lst.append(param["alpha"])
			num_pts_lst.append(param["num_points"])
		alphas = list(set(alpha_lst))
		alphas.sort()
		num_pts = list(set(num_pts_lst))
		num_pts.sort()
		N = len(alphas)
		M = len(num_pts)
				
		size = self.proteins[0].shape[0]

		# fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row',
		# 						gridspec_kw={'height_ratios': [1, 3],
		# 									'width_ratios': [1, 3]}
		# 						)
		# axs[0,0].remove()
		fig, ax = plt.subplots(figsize=(6,4))
		
		canvas = np.zeros((size*N, size*M))
		
		plot_num = 0 
		for i, alpha in enumerate(alphas):
			for j, num_points in enumerate(num_pts):
				for k, prot in enumerate(self.proteins):
					if (self.params[k]["alpha"] == alpha) and (self.params[k]["num_points"] == num_points):
						protein = prot
						break
				canvas[i*size:(i+1)*size, j*size:(j+1)*size] = prot
				
		plt.imshow(canvas, origin='lower', interpolation='nearest', resample=False, filternorm=False, cmap=plt.get_cmap('binary'))
		plt.xticks(ticks=[i*size + size/2 for i in range(M)], labels=['%d'%(num_points) for num_points in num_pts], fontsize=12)
		plt.yticks(ticks=[i*size + size/2 for i in range(N)], labels=['%.2f'%(alpha) for alpha in alphas], fontsize=12)
		plt.xlabel('Number of points', fontsize=16)
		plt.ylabel('Alpha', fontsize=16)
		ax.grid(b=False)
				
		def get_bars(val_lst):
			bar_values = list(set(val_lst))
			bar_values.sort()
			bar = []
			tick = []
			total = 0
			for i, value in enumerate(bar_values):
				total += val_lst.count(value)
			for i, value in enumerate(bar_values):
				bar.append(float(val_lst.count(value))/float(total))
				tick.append(i*size + size/2)
			return tick, bar
		
		
		divider = make_axes_locatable(ax)
		
		ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
		ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
		ax_histx.grid(b=False)
		ax_histy.grid(b=False)
		
		ax_histx.xaxis.set_tick_params(labelbottom=False)
		ax_histy.yaxis.set_tick_params(labelleft=False)
		a, b = get_bars(alpha_lst)
		ax_histy.barh(a,b, height=size*0.75)
		ax_histy.set_xlabel('Fraction', fontsize=16)

		a, b = get_bars(num_pts_lst)
		ax_histx.bar(a,b,width=size*0.75)
		ax_histx.set_ylabel('Fraction', fontsize=16)
		
		plt.tight_layout()
		plt.savefig(output_name)
