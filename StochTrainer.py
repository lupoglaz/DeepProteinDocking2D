import torch
from torch import optim
import torch.nn as nn
import numpy as np

from Models import EQScoringModelV2, EQDockModel
from Models import ProteinConv2D
from torchDataset import get_dataset_stream
from tqdm import tqdm
import random
from math import cos, sin

from matplotlib import pylab as plt

class StochTrainer:
	def __init__(self, model, optimizer, buffer, device='cuda', num_samples=10, weight=1.0, step_size=10.0, sample_steps=100):
		self.model = model
		self.optimizer = optimizer
		self.buffer = buffer

		self.num_samples = num_samples
		self.sample_steps = sample_steps
		self.weight = weight
		self.step_size = step_size
		self.device = device

		self.plot_idx = 0
		self.conv = ProteinConv2D()

	def requires_grad(self, flag=True):
		parameters = self.model.parameters()
		for p in parameters:
			p.requires_grad = flag

	def dock_spatial(self, rec_repr, lig_repr):
		translations = self.conv(rec_repr, lig_repr)
		
		batch_size = translations.size(0)
		num_features = translations.size(1)
		L = translations.size(2)

		translations = translations.view(batch_size, num_features, L*L)
		translations = translations.transpose(1,2).contiguous().view(batch_size*L*L, num_features)
		scores = self.model.scorer(translations).squeeze()
		scores = scores.view(batch_size, L, L)
		
		minval_y, ind_y = torch.min(scores, dim=2, keepdim=False)
		minval_x, ind_x = torch.min(minval_y, dim=1)
		x = ind_x
		y = ind_y[torch.arange(batch_size), ind_x]
		
		x -= int(L/2)
		y -= int(L/2)
		
		# plt.imshow(scores[0,:,:].detach().cpu(), cmap='magma')
		# plt.plot([y[0].item()], [x[0].item()], 'xb')
		# plt.show()
		# sys.exit()
		return torch.stack([x,y], dim=1).to(dtype=lig_repr.dtype, device=lig_repr.device)
				
	def rotate(self, repr, angle):
		alpha = angle.detach()
		T0 = torch.cat([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
		T1 = torch.cat([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
		R = torch.stack([T0, T1], dim=1)
		curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
		return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

	def langevin(self, neg_alpha, neg_dr, neg_rec, neg_lig, neg_idx, traces):
		noise_alpha = torch.zeros_like(neg_alpha)
		noise_dr = torch.zeros_like(neg_dr)

		self.requires_grad(False)
		self.model.eval()
		
		rec_feat = self.model.repr(neg_rec).tensor.detach()
		lig_feat = self.model.repr(neg_lig).tensor.detach()
		
		with torch.no_grad():
			rlig_feat = self.rotate(lig_feat, neg_alpha)
			neg_dr = self.dock_spatial(rec_feat, rlig_feat)
		
		neg_alpha.requires_grad_()
		neg_dr.requires_grad_()
		langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

		for k in range(self.sample_steps):
			langevin_opt.zero_grad()

			pos_repr, _, A = self.model.mult(rec_feat, lig_feat, neg_alpha, neg_dr)
			neg_out = self.model.scorer(pos_repr)
			neg_out.mean().backward()
			
			if len(traces) > 0:
				l=0
				for m, idx in enumerate(neg_idx):
					if idx.item() == self.plot_idx:
						traces[l].append( (neg_alpha[m].item(), neg_dr[m,0].item(), neg_dr[m,1].item()) )
						l+=1

			langevin_opt.step()
			
			neg_dr.data += noise_dr.normal_(0, 0.5)
			neg_alpha.data += noise_alpha.normal_(0, 0.05)

			neg_dr.data.clamp_(-neg_rec.size(2), neg_rec.size(2))
			neg_alpha.data.clamp_(-np.pi, np.pi)
		
		return neg_alpha.detach(), neg_dr.detach(), traces

	def step_stoch(self, data, epoch=None):
		receptor, ligand, translation, rotation, pos_idx = data
		
		pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		pos_alpha = rotation.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)*(np.pi/180.0)
		pos_dr = translation.to(device=self.device, dtype=torch.float32)

		batch_size = pos_rec.size(0)
		num_features = pos_rec.size(1)
		L = pos_rec.size(2)
		
		neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=self.num_samples)
		neg_rec = pos_rec.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size*self.num_samples, num_features, L, L)
		neg_lig = pos_lig.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size*self.num_samples, num_features, L, L)
		neg_idx = pos_idx.unsqueeze(dim=1).repeat(1, self.num_samples).view(batch_size*self.num_samples)
		neg_alpha = neg_alpha.view(batch_size*self.num_samples, -1)
		neg_dr = neg_dr.view(batch_size*self.num_samples, -1)
		
		traces = []
		for idx in neg_idx:
			if idx.item() == self.plot_idx:
				traces.append([])
		
		neg_alpha, neg_dr, traces = self.langevin(neg_alpha, neg_dr, neg_rec, neg_lig, neg_idx, traces=traces)
		
		if len(traces) > 0 and (not (epoch is None)):
			for m, idx in enumerate(pos_idx):
				if idx.item() == self.plot_idx:
					correct = (pos_alpha[m].item(), pos_dr[m,0].item(), pos_dr[m,1].item())

			with open(f"Log/traces_{epoch}.th", "wb") as fout:
				torch.save( (traces, correct), fout)


		self.requires_grad(True)
		self.model.train()
		self.model.zero_grad()
		
		pos_out = self.model(pos_rec, pos_lig, pos_alpha, pos_dr)
		L_p = (pos_out + self.weight * pos_out ** 2).mean()
		neg_out = self.model(neg_rec, neg_lig, neg_alpha, neg_dr)
		L_n = (-neg_out + self.weight * neg_out ** 2).mean()
		loss = L_p + L_n
		loss.backward()

		self.optimizer.step()
		self.buffer.push(pos_alpha, pos_dr, pos_idx)
		self.buffer.push(neg_alpha, neg_dr, neg_idx)
		
		return loss.item(),
	
