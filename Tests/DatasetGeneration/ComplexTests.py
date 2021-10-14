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
from DatasetGeneration import Complex, Protein

def scan_parameter(param, func, output_name='gap_score_param1.png', num_samples=10, name=""):
	import matplotlib.font_manager as font_manager
	from matplotlib import rcParams
	font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
	rcParams['font.family'] = 'Helvetica'
	f = plt.figure(figsize=(num_samples,len(param)+0.5))
	cell_size = 80
	canvas = np.zeros((cell_size*num_samples, cell_size*len(param)))
	plot_num = 0 
	for i, p in tqdm(enumerate(param), total=len(param)):
		for j in range(num_samples):
			rec = Protein.generateConcave(size=50, num_points=80, alpha=0.90)
			lig = Protein.generateConcave(size=50, num_points=80, alpha=0.90)
			cplx = func(rec, lig, p)
			canvas[j*cell_size:(j+1)*cell_size, i*cell_size:(i+1)*cell_size] = cplx.get_canvas(cell_size)
			
	plt.imshow(canvas.transpose(), origin='lower', interpolation='nearest', resample=False, filternorm=False, cmap='gist_heat_r')
	plt.xticks(ticks=[i*cell_size + cell_size/2 for i in range(num_samples)], labels=['%d'%(i+1) for i in range(num_samples)])
	plt.yticks(ticks=[i*cell_size + cell_size/2 for i in range(len(param))], labels=['%.2f'%i for i in param])
	plt.ylabel(name, fontsize=16)
	plt.xlabel('Sample number', fontsize=16)
	plt.tight_layout()
	plt.savefig(output_name)

if __name__=='__main__':
	rec = Protein.generateConcave(size=50, num_points=50)
	lig = Protein.generateConcave(size=50, num_points=50)
	cplx = Complex.generate(rec, lig)
	cplx.plot()
	plt.show()

	# scan_parameter(param=np.arange(0.1,0.75,0.05, dtype=np.float32), 
	# 				func=lambda x, y, p: Complex.generate(x, y, threshold=p),
	# 				num_samples=10, 
	# 				output_name='comp_overlap.png', name='Overlap')