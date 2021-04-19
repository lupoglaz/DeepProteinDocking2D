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

def load_log(filename):
	with open(filename) as fin:
		header = fin.readline()
		loss = [tuple([float(x) for x in line.split()]) for line in fin]
	return list(zip(*loss))

def plot(rec, lig, rot, trans):
	cplx = Complex(Protein(rec.numpy()), Protein(lig.numpy()), rot, trans)
	field = cplx.get_canvas(cell_size=90)
	plt.imshow(field)

def plot_eval(output_name, max_epoch=40, max_samples=5):
	fig = plt.figure(figsize=(20,4))
	camera = Camera(fig)
	for epoch in tqdm(range(max_epoch)):
		with open(f"Log/valid_{epoch}.th", "rb") as fin:
			log_data = torch.load( fin)
		num_targets = min(len(log_data), max_samples)
		
		cell_size = 100
		plot_image = np.zeros((2*cell_size, num_targets*cell_size))
		for i in range(num_targets):
			dict = log_data[i]
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
	animation.save(f'{output_name}.mp4')

def plot_dock(output_name, max_epoch=40):
	fig = plt.figure(figsize=(12,6))
	camera = Camera(fig)
	for epoch in tqdm(range(max_epoch)):
		with open(f"Log/valid_{epoch}.th", "rb") as fin:
			log_data = torch.load( fin)
		
		dict = log_data[0]
		angles, angle_scores = dict["rotations"]
			
		plt.subplot(1, 2, 1)
		plt.imshow(dict["translations"], cmap='plasma')
		plt.subplot(1, 2, 2)
		plt.plot(angles.cpu(), angle_scores)
		plt.tight_layout()	
		camera.snap()

	animation = camera.animate()
	animation.save(f'{output_name}.mp4')

def plot_traces(output_name, max_epoch=40):
	fig = plt.figure(figsize=(12,10))
	
	camera = Camera(fig)
	for epoch in tqdm(range(max_epoch)):
		with open(f"Log/traces_{epoch}.th", "rb") as fin:
			traces, correct = torch.load( fin)
		ca, cx, cy = correct
		
		ax = fig.gca(projection='3d')
		for trace in traces:
			angles, x, y = zip(*trace)
			ax.plot(x, y, angles, label='parametric curve')
		ax.scatter([cx], [cy], [ca], s=20, c="r")
		# print(correct)
		# sys.exit()
		camera.snap()
		# plt.show()

	animation = camera.animate()
	# plt.show()
	animation.save(f'{output_name}.mp4')

def plot_losses(output_name):
	f = plt.figure(figsize =(12,6))
	train = load_log('Log/log_train_scoring_v2.txt')
	valid = load_log('Log/log_valid_scoring_v2.txt')
	plt.subplot(1,2,1)
	plt.plot(train[1], label='train')
	plt.ylim([-0.6,0.2])

	plt.subplot(1,2,2)
	plt.plot(valid[1], label='valid')
	
	plt.legend()
	# plt.show()
	plt.savefig(f'{output_name}.png')

if __name__=='__main__':
	max_epoch = 100
	# plot_dock("dock", max_epoch=max_epoch)
	plot_eval("eval_anim", max_epoch=max_epoch)
	plot_losses("losses")

	# plot_traces("traces", max_epoch=max_epoch)