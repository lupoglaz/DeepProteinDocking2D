import os 
import sys 
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import pylab as plt
from DatasetGeneration import Complex, Protein
from celluloid import Camera
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

class Logger:
	def __init__(self, log_dir):
		assert log_dir.exists()
		self.log_dir = log_dir
		self.step = 0
		self.epoch = 0

	@classmethod
	def new(cls, log_dir):
		log_dir.mkdir(exist_ok=True, parents=True)
		with open(log_dir/Path('train.txt'), 'w') as fout:
			fout.write('Step\tLoss\n')
		with open(log_dir/Path('valid.txt'), 'w') as fout:
			fout.write('Step\tLoss\n')
		
		return cls(log_dir)

	def log_train(self, loss):
		with open(self.log_dir/Path('train.txt'), 'a') as fout:
			fout.write('%d\t%f\n'%(self.step, loss))
		self.step += 1

	def log_valid(self, loss):
		with open(self.log_dir/Path('valid.txt'), 'a') as fout:
			fout.write('%d\t%f\n'%(self.step, loss))
	
	def log_valid_inter(self, Accuracy, Precision, Recall):
		with open(self.log_dir/Path('valid.txt'), 'a') as fout:
			fout.write('%d\t%f\t%f\t%f\n'%(self.step, Accuracy, Precision, Recall))

	def log_data(self, data):
		with open(self.log_dir/Path(f'valid_{self.epoch}.th'), "wb") as fout:
			torch.save(data, fout)
		self.epoch += 1

	def plot_losses(self, output_name=None, average_num=5, xlabel='Iteration', ylabel='RMSD', coupled=False):
		import matplotlib.font_manager as font_manager
		from matplotlib import rcParams
		# font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
		# rcParams['font.family'] = 'Helvetica'
		def load_log(filename):
			with open(filename) as fin:
				header = fin.readline().split('\t')
				loss = [tuple([float(x) for x in line.split()]) for line in fin]
			data = [np.array(x) for x in list(zip(*loss))]
			return data, header
		
		def moving_average(x, w):
			return np.convolve(x, np.ones(w), 'valid') / w

		loss_train, header_train = load_log(self.log_dir/Path('train.txt'))
		loss_valid, header_valid = load_log(self.log_dir/Path('valid.txt'))
		
		train_x = moving_average(loss_train[0], average_num)
		train_y = moving_average(loss_train[1], average_num)
		
		if coupled:
			f = plt.figure(figsize =(6,6))
			plt.plot(train_x, train_y, label='Training')
			plt.plot(loss_valid[0], loss_valid[1], label='Validation')
			plt.xlabel(xlabel, fontsize=18)
			plt.ylabel(ylabel, fontsize=18)
			plt.legend(fontsize=18)
		else:
			f = plt.figure(figsize =(12,6))
			plt.subplot(1,2,1)
			plt.plot(train_x, train_y, label='train')
			plt.ylim([np.min(train_y)-0.1, np.median(train_y)+5.0])
			plt.subplot(1,2,2)
			plt.plot(loss_valid[0], loss_valid[1], label='valid')
			plt.legend(fontsize=18)

		if output_name is None:
			plt.show()
		else:
			plt.savefig(f'{output_name}.png')

	def plot_dock(self, output_name, max_epoch=40):
		fig, axs = plt.subplots(1, 2, figsize=(12,6))
		camera = Camera(fig)
		for epoch in tqdm(range(1)):
			with open(self.log_dir/Path(f'valid_{epoch}.th'), "rb") as fin:
				log_data = torch.load( fin)

			dict = log_data[0]
			if (not("rotations" in dict.keys())) or (not("translations" in dict.keys())):
				return
			angles, angle_scores = dict["rotations"]

			axs[0].imshow(dict["translations"], cmap='plasma')
			axs[1].plot(angles.cpu(), angle_scores)
			plt.tight_layout()
			camera.snap()

		animation = camera.animate()
		animation.save(output_name.with_suffix('.mp4').as_posix())

	def plot_eval(self, output_name, max_epoch=40, max_samples=5):
		fig = plt.figure(figsize=(20,4))
		camera = Camera(fig)
		for epoch in tqdm(range(1)):
			with open(self.log_dir/Path(f'valid_{epoch}.th'), "rb") as fin:
				log_data = torch.load( fin)
			num_targets = min(len(log_data), max_samples)

			cell_size = 100
			plot_image = np.zeros((2*cell_size, num_targets*cell_size))
			for i in range(num_targets):
				dict = log_data[i]
				if (not("receptors" in dict.keys())) or (not("ligands" in dict.keys())):
					return
				rec = dict["receptors"][0,0,:,:]
				lig = dict["ligands"][0,0,:,:]
				cplx = Complex(Protein(rec.numpy()), Protein(lig.numpy()), dict["rotation"].item(), dict["translation"].squeeze().numpy())
				plot_image[:cell_size, i*cell_size:(i+1)*cell_size] = cplx.get_canvas(cell_size=cell_size)

				rec = dict["receptors"][0,0,:,:]
				lig = dict["ligands"][0,0,:,:]
				cplx = Complex(Protein(rec.numpy()), Protein(lig.numpy()), dict["pred_rotation"].item(), dict["pred_translation"].numpy())
				plot_image[cell_size:, i*cell_size:(i+1)*cell_size] = cplx.get_canvas(cell_size=cell_size)

			plt.imshow(plot_image)
			# plt.show()
			camera.snap()

		animation = camera.animate()
		animation.save(output_name.with_suffix('.mp4').as_posix())

	def plot_losses_int(self, output_name=None, average_num=5):
		def load_log(filename):
			with open(filename) as fin:
				header = fin.readline()
				loss = [tuple([float(x) for x in line.split()]) for line in fin]
			data = [np.array(x) for x in list(zip(*loss))]
			return data
		
		def moving_average(x, w):
			return np.convolve(x, np.ones(w), 'valid') / w

		loss_train = load_log(self.log_dir/Path('train.txt'))
		loss_valid = load_log(self.log_dir/Path('valid.txt'))
		
		train_x = moving_average(loss_train[0], average_num)
		train_y = moving_average(loss_train[1], average_num)
		f = plt.figure(figsize =(12,6))
		plt.subplot(1,2,1)
		plt.plot(train_x, train_y, label='train')

		plt.subplot(1,2,2)
		plt.plot(loss_valid[0], loss_valid[1], label='accuracy')
		plt.plot(loss_valid[0], loss_valid[2], label='precision')
		plt.plot(loss_valid[0], loss_valid[3], label='recall')
		plt.legend()

		if output_name is None:
			plt.show()
		else:
			plt.savefig(f'{output_name}.png')