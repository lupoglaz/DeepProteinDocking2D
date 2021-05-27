import os 
import sys 
import argparse
from pathlib import Path
from Logger import Logger

import torch
from DockingTrainer import DockingTrainer
from Models import EQScoringModel
from torchDataset import get_docking_stream

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import pylab as plt
from matplotlib import colors

def plot_representations(experiments, output_file='Representations.png'):
	fig, axs = plt.subplots(2, 3, figsize=(12,8), sharey=True, constrained_layout=True)
	# fig.subplots_adjust(wspace=0)
	

	test_stream = get_docking_stream('DatasetGeneration/docking_data_test.pkl', batch_size=1, max_size=None)
	for data in test_stream:
		receptor, ligand, translation, rotation, pos_idx = data
		receptor = receptor.to(device='cuda', dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device='cuda', dtype=torch.float32).unsqueeze(dim=1)
		break

	model = EQScoringModel().to(device='cuda')
	trainer = DockingTrainer(model, None, type='pos')
	trainer.load_checkpoint(Path('Log')/Path(experiments[0])/Path('dock_ebm.th'))
	repr_0 = model.repr(receptor).tensor.detach().cpu().clone()
	trainer.load_checkpoint(Path('Log')/Path(experiments[1])/Path('dock_ebm.th'))
	repr_1 = model.repr(receptor).tensor.detach().cpu().clone()
	reprs = torch.cat([repr_0, repr_1], dim=0)
	min, max = torch.min(reprs).item(), torch.max(reprs).item()
	norm = colors.TwoSlopeNorm(vcenter=0.0)
	
	axs[0,0].set_title('Input', fontsize=16)
	a = axs[0,0].imshow(receptor[0,0,:,:].cpu(), cmap = 'binary')
	axs[0,1].set_title('Brute force learning', fontsize=16)
	b = axs[0,1].imshow(repr_0[0,0,:,:], cmap='PRGn', norm=norm)
	c = axs[1,1].imshow(repr_0[0,1,:,:], cmap='PRGn', norm=norm)
			
	repr = model.repr(receptor).tensor.detach().cpu()
	axs[0,2].set_title('Energy-based learning', fontsize=16)
	e = axs[0,2].imshow(repr_1[0,0,:,:], cmap='PRGn', norm=norm)
	f = axs[1,2].imshow(repr_1[0,1,:,:], cmap='PRGn', norm=norm)

	axs[1, 0].remove()
	
	fig.colorbar(a, ax=axs[0,0], shrink=0.6, location='bottom')
	# fig.colorbar(b, ax=axs[0,1], shrink=0.6, location='bottom')
	fig.colorbar(c, ax=axs[1,1], shrink=0.6, location='bottom')
	# fig.colorbar(d, ax=axs[1,0], shrink=0.6, location='bottom')
	# fig.colorbar(e, ax=axs[1,1], shrink=0.6, location='bottom')
	fig.colorbar(f, ax=axs[1,2], shrink=0.6, location='bottom')
	if output_file is None:
		plt.show()
	else:
		plt.savefig(output_file)


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	parser.add_argument('-experiment', default='DebugDocking', type=str)
	parser.add_argument('-interaction', action='store_const', const=lambda:'interaction', dest='type')
	parser.add_argument('-docking', action='store_const', const=lambda:'docking', dest='type')
	parser.add_argument('-max_epoch', default=100, type=int)
	args = parser.parse_args()

	LOG_DIR = Path('Log')/Path(args.experiment)
	RES_DIR = Path('Results')/Path(args.experiment)
	RES_DIR.mkdir(parents=True, exist_ok=True)

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(0)

	logger = Logger(LOG_DIR)
	if args.type() == 'interaction':
		logger.plot_losses_int(RES_DIR/Path("losses"), average_num=30)
	else:
		# pass
		max_epoch = 100
		logger.plot_losses(RES_DIR/Path("losses"), coupled=True)
		logger.plot_dock(RES_DIR/Path("dock"), max_epoch=max_epoch)
		logger.plot_eval(RES_DIR/Path("eval_anim"), max_epoch=max_epoch)
		plot_representations(['BFDocking', 'EBMDocking'], output_file=RES_DIR/Path("comp.png"))
	