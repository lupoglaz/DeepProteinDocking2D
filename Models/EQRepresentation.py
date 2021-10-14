import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from e2cnn import gspaces
from e2cnn import nn as e2nn

def init_weights(module):
	if isinstance(module, e2nn.R2Conv):
		# e2nn.init.deltaorthonormal_init(module.weights, module.basisexpansion)
		pass

class EQRepresentation(nn.Module):
	def __init__(self, bias=False):
		super(EQRepresentation, self).__init__()

		r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
		self.feat_type_in = e2nn.FieldType(r2_act, [r2_act.irreps['irrep_0']])
		self.feat_type_hid = e2nn.FieldType(r2_act, 16*[r2_act.irreps['irrep_0']]+4*[r2_act.irreps['irrep_1']]+2*[r2_act.irreps['irrep_2']])
		self.feat_type_out = e2nn.FieldType(r2_act, 2*[r2_act.trivial_repr])
		
		self.repr = nn.Sequential(
			e2nn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=5, padding=2, bias=bias),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=bias),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),

			e2nn.R2Conv(self.feat_type_hid, self.feat_type_hid, kernel_size=5, padding=2, bias=bias),
			e2nn.NormNonLinearity(self.feat_type_hid, bias=False),
			
			e2nn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size=5, padding=2, bias=bias),
		)
		with torch.no_grad():
			self.repr.apply(init_weights)

	def forward(self, protein):
		x = e2nn.GeometricTensor(protein, self.feat_type_in)
		return self.repr(x)

class EQRepresentationSid(nn.Module):

	def __init__(self):
		super(EQRepresentationSid, self).__init__()
		
		self.scal = 1
		self.vec = 3
		self.SO2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=4)
		self.feat_type_in1 = e2nn.FieldType(self.SO2, 1 * [self.SO2.trivial_repr])
		self.feat_type_out1 = e2nn.FieldType(self.SO2, self.scal * [self.SO2.irreps['irrep_0']] + self.vec * [self.SO2.irreps['irrep_1']])
		self.feat_type_out_final = e2nn.FieldType(self.SO2, 1 * [self.SO2.irreps['irrep_0']] + 1 * [self.SO2.irreps['irrep_1']])

		self.kernel = 5
		self.pad = self.kernel//2
		self.stride = 1
		self.dilation = 1

		self.netSE2 = e2nn.SequentialModule(
			e2nn.R2Conv(self.feat_type_in1, self.feat_type_out1, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad , bias=False),
			e2nn.NormNonLinearity(self.feat_type_out1, function='n_relu', bias=False),
			e2nn.R2Conv(self.feat_type_out1, self.feat_type_out_final, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad, bias=False),
			e2nn.NormNonLinearity(self.feat_type_out_final, function='n_relu', bias=False),
			e2nn.NormPool(self.feat_type_out_final)
		)

	def forward(self, protein):
		proteinT = e2nn.GeometricTensor(protein, self.feat_type_in1)
		feat = self.netSE2(proteinT)
		return feat