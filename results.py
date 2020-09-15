import os 
import sys 
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import pylab as plt
from celluloid import Camera
from DatasetGeneration import rotate_ligand
import torch
import numpy as np
from tqdm import tqdm

def load_log(filename):
	with open(filename) as fin:
		header = fin.readline()
		loss = [tuple([float(x) for x in line.split()]) for line in fin]
	return list(zip(*loss))

def plot(rec, lig, rot, trans):
	box_size = int(rec.size(1))
	field_size = box_size*3
	field = np.zeros( (field_size, field_size) )
	rot_lig = rotate_ligand(lig[:,:].numpy(), rot[0].item()*180.0/np.pi)
	dx = int(trans[0].item())
	dy = int(trans[1].item())
	# print(dx, dy, rot[0].item()*180.0/np.pi)

	field[ 	int(field_size/2 - box_size/2): int(field_size/2 + box_size/2),
				int(field_size/2 - box_size/2): int(field_size/2 + box_size/2)] += rec.numpy()
	
	field[  int(field_size/2 - box_size/2 + dx): int(field_size/2 + box_size/2 + dx),
			int(field_size/2 - box_size/2 + dy): int(field_size/2 + box_size/2 + dy) ] += 2*rot_lig

	plt.imshow(field)

def plot_eval(output_name, max_epoch=40):
	fig = plt.figure(figsize=(20,4))
	camera = Camera(fig)
	for epoch in tqdm(range(max_epoch)):
		with open(f"Log/valid_{epoch}.th", "rb") as fin:
			dict = torch.load( fin)
		num_targets = dict["receptors"].size(0)
		for i in range(num_targets):
			plt.subplot(2, num_targets, i + 1)
			plot(dict["receptors"][i,0,:,:], dict["ligands"][i,0,:,:], dict["rotation"][i], dict["translation"][i,:])
			plt.subplot(2, num_targets, num_targets + i + 1)
			plot(dict["receptors"][i,0,:,:], dict["ligands"][i,0,:,:], dict["pred_rotation"][i], dict["pred_translation"][i,:])
		
		plt.tight_layout()	
		camera.snap()

	animation = camera.animate()
	animation.save(f'{output_name}.mp4')

def plot_dock(output_name, max_epoch=40):
	fig = plt.figure(figsize=(12,6))
	camera = Camera(fig)
	for epoch in tqdm(range(max_epoch)):
		with open(f"Log/valid_{epoch}.th", "rb") as fin:
			dict = torch.load( fin)
	
		angles, angle_scores = dict["rotations"]
			
		plt.subplot(1, 2, 1)
		plt.imshow(dict["translations"])
		plt.subplot(1, 2, 2)
		plt.plot(angles, angle_scores)
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
		camera.snap()
		# plt.show()

	animation = camera.animate()
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
	plot_dock("dock", max_epoch=100)
	plot_traces("traces", max_epoch=100)
	plot_eval("eval_anim", max_epoch=100)
	plot_losses("losses")