import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from e2cnn import gspaces
from e2cnn import nn as e2nn

def init_weights(module):
	if isinstance(module, e2nn.R2Conv):
		e2nn.init.deltaorthonormal_init(module.weights, module.basisexpansion)
	elif isinstance(module, torch.nn.Linear):
		module.bias.data.zero_()

class EQRepresentation(nn.Module):
	def __init__(self):
		super(EQRepresentation, self).__init__()

		r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
		self.feat_type_in = e2nn.FieldType(r2_act, [r2_act.irreps['irrep_0']])
		self.feat_type_hid = e2nn.FieldType(r2_act, 16*[r2_act.irreps['irrep_0']]+4*[r2_act.irreps['irrep_1']]+2*[r2_act.irreps['irrep_2']])
		self.feat_type_out = e2nn.FieldType(r2_act, 4*[r2_act.trivial_repr])
		
		self.repr = nn.Sequential(
			e2nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=False),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),
			
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size=5, padding=2, bias=False),
		)
		with torch.no_grad():
			self.repr.apply(init_weights)

	def forward(self, protein):
		x = e2nn.GeometricTensor(protein, self.feat_type_in)
		return self.repr(x)