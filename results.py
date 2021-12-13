import os 
import sys 
import argparse
from pathlib import Path
from Logger import Logger

import torch
from DockingTrainer import DockingTrainer
from Models import EQScoringModel
from Models import EQRepresentation
from torchDataset import get_docking_stream

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import pylab as plt
from matplotlib import colors

def plot_representations(experiments, output_file='Representations.png'):
	from DatasetGeneration import Protein
	import matplotlib.font_manager as font_manager
	from matplotlib import rcParams
	font_manager.fontManager.addfont('/home/lupoglaz/.fonts/Helvetica.ttf')
	rcParams['font.family'] = 'Helvetica'
			
	test_stream = get_docking_stream('Data/docking_data_test.pkl', batch_size=1, max_size=None)
	for data in test_stream:
		receptor, ligand, translation, rotation, pos_idx = data
		receptor = receptor.unsqueeze(dim=1).to(dtype=torch.float32)
		ligand = ligand.unsqueeze(dim=1).to(dtype=torch.float32)
		break

	model = EQScoringModel(EQRepresentation())
	trainer = DockingTrainer(model, None, type='pos')
	trainer.load_checkpoint(Path('Data')/Path(experiments[0])/Path('model.th'))
	repr_0 = model.repr(ligand).tensor.detach().cpu().clone()
	A = model.scorer[0].weight.view(2,2)
	eigvals, V = torch.linalg.eig(A)
	V = V.real
	rv01 = V[0,0] * repr_0[0,0,:,:] + V[1,0]*repr_0[0,1,:,:]
	rv02 = V[0,1] * repr_0[0,0,:,:] + V[1,1]*repr_0[0,1,:,:]
	repr_0 = torch.stack([rv01, rv02],dim=0).unsqueeze(dim=0).detach()
	print(eigvals)

	trainer.load_checkpoint(Path('Data')/Path(experiments[1])/Path('model.th'))
	repr_1 = model.repr(ligand).tensor.detach().cpu().clone()
	A = model.scorer[0].weight.view(2,2)
	eigvals, V = torch.linalg.eig(A)
	V = V.real
	rv11 = V[0,0] * repr_1[0,0,:,:] + V[1,0]*repr_1[0,1,:,:]
	rv12 = V[0,1] * repr_1[0,0,:,:] + V[1,1]*repr_1[0,1,:,:]
	repr_1 = torch.stack([rv11, rv12],dim=0).unsqueeze(dim=0).detach()
	print(eigvals)
	
	# sys.exit()
	reprs = torch.cat([repr_0, repr_1], dim=0)
	min, max = torch.min(reprs).item(), torch.max(reprs).item()
	norm = colors.TwoSlopeNorm(vcenter=0.0)
	# fig, axs = plt.subplots(2, 3, figsize=(10,5), constrained_layout=True)
	fig, axs = plt.subplots(3, 2, figsize=(8,12), sharey=True, constrained_layout=True)
	fig.subplots_adjust(wspace=0)
	fig.subplots_adjust(hspace=1.0)

	# axs[0,0].set_title('Input', fontsize=22)
	axs[0,0].set_title('Input', fontsize=22)
	axs[0,1].set_title('Boundary', fontsize=22)
	# plt.figtext(0.55, 0.985,"Input", va="center", ha="center", size=22)
	a = axs[0,0].imshow(ligand[0,0,:,:].cpu(), cmap = 'binary')
	prot = Protein(ligand[0,0,:,:].to(dtype=torch.double, device='cpu').numpy())
	prot.make_boundary()
	# axs[1,0].set_title('Boundary', fontsize=22)
	a1 = axs[0,1].imshow(prot.boundary, cmap = 'binary')

	# axs[0,1].set_title('BFDocking', fontsize=22)
	axs[1,0].set_title(' ', fontsize=22)
	plt.figtext(0.55,0.65,"BF Docking", va="center", ha="center", size=22)

	b = axs[1,0].imshow(repr_0[0,1,:,:], cmap='coolwarm', norm=norm)
	c = axs[1,1].imshow(repr_0[0,0,:,:], cmap='coolwarm', norm=norm)
			
	repr = model.repr(receptor).tensor.detach().cpu()
	# axs[0,2].set_title('BFInteraction', fontsize=22)
	axs[2,0].set_title(' ', fontsize=22)
	plt.figtext(0.55, 0.315,"BF Interaction", va="center", ha="center", size=22)
	e = axs[2,0].imshow(repr_1[0,0,:,:], cmap='coolwarm', norm=norm)
	f = axs[2,1].imshow(repr_1[0,1,:,:], cmap='coolwarm', norm=norm)

	# axs[0, 1].remove()
	
	
	fig.colorbar(a, ax=axs[0,0], shrink=0.8, location='left')
	fig.colorbar(c, ax=axs[1,0], shrink=0.8, location='left')
	fig.colorbar(f, ax=axs[2,0], shrink=0.8, location='left')
	# fig.colorbar(a1, ax=axs[:,0], shrink=0.5)
	# fig.colorbar(c, ax=axs[:,1], shrink=0.5)
	# fig.colorbar(f, ax=axs[:,2], shrink=0.5)
	plt.tight_layout()
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

	plot_representations(['BFDocking', 'BFInteraction'], Path("comp_poster.png"))
	sys.exit()

	LOG_DIR = Path('Log')/Path(args.experiment)
	RES_DIR = Path('Results')/Path(args.experiment)
	RES_DIR.mkdir(parents=True, exist_ok=True)

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(0)

	logger = Logger(LOG_DIR)
	if args.type() == 'interaction':
		logger.plot_losses_int(RES_DIR/Path("losses"), average_num=50)
	else:
		# pass
		max_epoch = args.max_epoch
		logger.plot_losses(RES_DIR/Path("losses"), coupled=True)
		logger.plot_dock(RES_DIR/Path("dock"), max_epoch=max_epoch)
		logger.plot_eval(RES_DIR/Path("eval_anim"), max_epoch=max_epoch)
		# plot_representations(['BFDocking', 'EBMDocking'], output_file=RES_DIR/Path("comp.png"))
	