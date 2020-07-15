import torch
from torch import nn
from torch.autograd import Function
import numpy as np

from .convolution import ProteinConv2D

class SpatialDockModel(nn.Module):
	def __init__(self, num_features=1, prot_field_size=50):
		super(SpatialDockModel, self).__init__()
		self.prot_field_size = prot_field_size
				
		self.convolution = ProteinConv2D()
		
		self.repr = nn.Sequential(
			nn.Conv2d(num_features, 2, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),

			nn.Conv2d(2, 4, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),

			nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),

			nn.Conv2d(16, 4, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
		)

		self.reduce = nn.Sequential(
			nn.Conv2d(18, 32, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=1),
		)

		self.get_translation = nn.Sequential(
			nn.Linear(16, 2),
		)

	def forward(self, receptor, ligand):
		rec_feat = self.repr(receptor.unsqueeze(dim=1))
		lig_feat = self.repr(ligand.unsqueeze(dim=1))
		translations = self.convolution(rec_feat, lig_feat)

		with torch.no_grad():
			box_size = translations.size(2)
			batch_size = translations.size(0)
			coords = torch.linspace(-self.prot_field_size, self.prot_field_size, box_size).to(device=translations.device)
			x = coords.repeat(batch_size, box_size, 1)
			y = x.transpose(1,2)
			coords = torch.stack([x,y], dim=1)

		translations = torch.cat([translations, coords], dim=1)

		output = self.reduce(translations).squeeze()
		
		return self.get_translation(output), output