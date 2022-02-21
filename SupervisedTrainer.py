import torch
from torch import optim
import torch.nn as nn
import numpy as np

# from .Models import ProteinConv2D
from .Models.Convolution.ProteinConvolution2D import ProteinConv2D

from tqdm import tqdm
import random
from math import cos, sin
import numpy as np

from matplotlib import pylab as plt

class RMSDLoss(nn.Module):
	def __init__(self):
		super(RMSDLoss, self).__init__()

	def get_XC(self, protein):
		"""
		Analog of inertia tensor and center of mass for rmsd calc. Return 2/W * X and C
		"""
		batch_size = protein.size(0)
		size = protein.size(2)
		X = torch.zeros(batch_size, 2, 2, device=protein.device)
		C = torch.zeros(batch_size, 2, device=protein.device)
		x_i = (torch.arange(size).unsqueeze(dim=0).to(device=protein.device) - size/2.0).repeat(size, 1)
		y_i = (torch.arange(size).unsqueeze(dim=1).to(device=protein.device) - size/2.0).repeat(1, size)
		x_i = x_i.unsqueeze(dim=0).repeat(batch_size, 1, 1)
		y_i = y_i.unsqueeze(dim=0).repeat(batch_size, 1, 1)
		
		mask = (protein > 0.5).to(dtype=torch.float32).squeeze()
		W = torch.sum(mask) + 1E-5
		x_i = x_i*mask
		y_i = y_i*mask
		#Inertia tensor
		X[:,0,0] = torch.sum(torch.sum(x_i*x_i, dim=-1), dim=-1)
		X[:,1,1] = torch.sum(torch.sum(y_i*y_i, dim=-1), dim=-1)
		X[:,0,1] = torch.sum(torch.sum(x_i*y_i, dim=-1), dim=-1)
		X[:,1,0] = torch.sum(torch.sum(y_i*x_i, dim=-1), dim=-1)
		#Center of mass
		C[:,0] = torch.sum(torch.sum(x_i, dim=-1), dim=-1)
		C[:,1] = torch.sum(torch.sum(x_i, dim=-1),dim=-1)
		return 2.0*X/W, C/W

	def forward(self, ligand, translation1, rotation1, translation2, rotation2):
		with torch.no_grad():
			X, C = self.get_XC(ligand)
		T = translation1 - translation2
		
		T0 = torch.stack([torch.cos(rotation1), -torch.sin(rotation1)], dim=1)
		T1 = torch.stack([torch.sin(rotation1), torch.cos(rotation1)], dim=1)
		R1 = torch.stack([T0, T1], dim=1)

		T0 = torch.stack([torch.cos(rotation2), -torch.sin(rotation2)], dim=1)
		T1 = torch.stack([torch.sin(rotation2), torch.cos(rotation2)], dim=1)
		R2 = torch.stack([T0, T1], dim=1)

		R = R2.transpose(1,2) @ R1
		I = torch.diag(torch.ones(2, device=ligand.device)).unsqueeze(dim=0).repeat(ligand.size(0),1,1)
		#RMSD
		rmsd = torch.sum(T*T)
		rmsd = rmsd + torch.sum((I-R)*X, dim=(1,2))
		rmsd = rmsd + 2.0*torch.sum(torch.sum(T.unsqueeze(dim=2) * (R1-R2), dim=1) * C, dim=1)
		return torch.sqrt(rmsd)


class SupervisedTrainer:
	def __init__(self, model, optimizer, device='cuda', type='pos'):
		self.model = model
		self.optimizer = optimizer
		self.device = device
		self.type = type
		if self.type=='int':
			self.loss = nn.BCELoss()
		elif self.type=='pos':
			self.loss = RMSDLoss()
		else:
			raise(Exception('Type unknown:', type))

	def load_checkpoint(self, path):
		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		checkpoint = torch.load(path)
		raw_model.load_state_dict(checkpoint)

	def rotate(self, repr):
		with torch.no_grad():
			alpha = torch.rand(repr.size(0), dtype=torch.float32, device=repr.device)*2.0*np.pi
			T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
			T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
			R = torch.stack([T0, T1], dim=1)
			curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
			return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

	def step(self, data, epoch=None):
		if self.type == 'int':
			receptor, ligand, target = data
			receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			receptor = self.rotate(receptor)
			ligand = self.rotate(ligand)
			target = target.to(device=self.device).unsqueeze(dim=1)
		elif self.type == 'pos':
			receptor, ligand, translation, rotation, pos_idx = data
			receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
			rotation = rotation.to(device=self.device, dtype=torch.float32)
			translation = translation.to(device=self.device, dtype=torch.float32)	

		self.model.train()
		self.model.zero_grad()
		rec, lig = self.model(receptor, ligand)
		if self.type == 'int':
			pred = self.model.sigmoid(self.model.fc_int(torch.cat([rec,lig], dim=1)))
			loss = self.loss(pred, target)
		elif self.type == 'pos':
			pred = self.model.fc_pos(torch.cat([rec,lig], dim=1))
			loss = torch.mean(self.loss(ligand, pred[:,:2], pred[:,-1], translation, rotation))	

		loss.backward()
		self.optimizer.step()
		log_dict = {"Loss": loss.item()}
		return log_dict

	def eval(self, data, threshold=0.5):
		if self.type == 'int':
			receptor, ligand, target = data
		else:
			receptor, ligand, translation, rotation, pos_idx = data
			translation = translation.to(device=self.device, dtype=torch.float32)
			rotation = rotation.to(device=self.device, dtype=torch.float32)

		receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		self.model.eval()
		with torch.no_grad():
			rec, lig = self.model(receptor, ligand)
			if self.type == 'int':
				pred = self.model.sigmoid(self.model.fc_int(torch.cat([rec,lig], dim=1)))
				TP = 0
				FP = 0
				TN = 0
				FN = 0
				for i in range(pred.size(0)):
					p = pred[i].item()
					a = target[i].item()
					if p>=threshold and a>=threshold:
						TP += 1
					elif p>=threshold and a<threshold:
						FP += 1
					elif p<threshold and a>=threshold:
						FN += 1
					elif p<threshold and a<threshold:
						TN += 1
				
				log_dict = {"TP": TP,
							"FP": FP,
							"FN": FN,
							"TN": TN}
			else:
				pred = self.model.fc_pos(torch.cat([rec,lig], dim=1))
				loss = torch.mean(self.loss(ligand, pred[:,:2], pred[:,-1], translation, rotation))
				log_dict = {"Loss": loss.item(),
							"Translation": pred[:,:2],
							"Rotation": pred[:,-1]}
			return log_dict
