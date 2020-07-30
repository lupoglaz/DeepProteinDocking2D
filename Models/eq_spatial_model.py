import torch
from torch import nn
from torch.autograd import Function
import numpy as np

from .convolution import ProteinConv2D
from e2cnn import gspaces
from e2cnn import nn as e2nn

class EQRepresentation(nn.Module):
	def __init__(self):
		super(EQRepresentation, self).__init__()

		r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
		self.feat_type_in = e2nn.FieldType(r2_act, [r2_act.trivial_repr])
		self.feat_type_hid = e2nn.FieldType(r2_act, 8*[r2_act.trivial_repr])
		self.feat_type_out = e2nn.FieldType(r2_act, 8*[r2_act.trivial_repr])
		
		self.repr = nn.Sequential(
			e2nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=3, padding=1),
			e2nn.ReLU(self.feat_type_hid),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=3, padding=1),
			e2nn.ReLU(self.feat_type_hid),
			
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size=3, padding=1),
		)

	def forward(self, protein):
		x = e2nn.GeometricTensor(protein, self.feat_type_in)
		return self.repr(x)

class EQSpatialDockModel(nn.Module):
	def __init__(self, num_features=1, prot_field_size=50):
		super(EQSpatialDockModel, self).__init__()
		self.prot_field_size = prot_field_size
				
		self.convolution = ProteinConv2D()
		self.repr = EQRepresentation()

		r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
		self.feat_type_in = e2nn.FieldType(r2_act, 6*[r2_act.trivial_repr])
		self.feat_type_hid = e2nn.FieldType(r2_act, 32*[r2_act.trivial_repr])
		self.feat_type_out = e2nn.FieldType(r2_act, 16*[r2_act.trivial_repr])

		self.reduce = nn.Sequential(
			e2nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=3, padding=1),
			e2nn.ReLU(self.feat_type_hid),
			e2nn.PointwiseMaxPoolAntialiased(self.feat_type_hid, kernel_size=5, stride=3, padding=0),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=3, padding=1),
			e2nn.ReLU(self.feat_type_hid),
			e2nn.PointwiseMaxPoolAntialiased(self.feat_type_hid, kernel_size=5, stride=3, padding=0),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size=3, padding=1),
			e2nn.ReLU(self.feat_type_out),
			e2nn.PointwiseMaxPoolAntialiased(self.feat_type_out, kernel_size=7, stride=5, padding=0),
		)
		self.get_translation = nn.Sequential(
			nn.Linear(16, 2),
		)

	def forward(self, receptor, ligand):
		rec_feat = self.repr(receptor.unsqueeze(dim=1))
		lig_feat = self.repr(ligand.unsqueeze(dim=1))
		translations = self.convolution(rec_feat.tensor, lig_feat.tensor)

		with torch.no_grad():
			box_size = translations.size(2)
			batch_size = translations.size(0)
			coords = torch.linspace(-self.prot_field_size, self.prot_field_size, box_size).to(device=translations.device)
			x = coords.repeat(batch_size, box_size, 1)
			y = x.transpose(1,2)
			coords = torch.stack([x,y], dim=1)

		translations = e2nn.GeometricTensor(torch.cat([translations, coords], dim=1), self.feat_type_in)
		output = self.reduce(translations).tensor.squeeze()
		return self.get_translation(output), output