import torch
from torch import nn
from torch.autograd import Function
from torch.nn.functional import affine_grid, grid_sample
import numpy as np

from .convolution import EQProteinConv2D
from e2cnn import gspaces
from e2cnn import nn as e2nn

class EQRepresentation(nn.Module):
	def __init__(self):
		super(EQRepresentation, self).__init__()

		r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
		self.feat_type_in = e2nn.FieldType(r2_act, [r2_act.trivial_repr])
		self.feat_type_hid_scalar = e2nn.FieldType(r2_act, 8*[r2_act.trivial_repr])
		self.feat_type_hid_vector = e2nn.FieldType(r2_act, 2*[r2_act.irrep(1)])
		self.feat_type_out = e2nn.FieldType(r2_act, 3*[r2_act.trivial_repr])

		relu = e2nn.ReLU(self.feat_type_hid_scalar)
		norm_relu = e2nn.NormNonLinearity(self.feat_type_hid_vector)
		nonlinearity_hid = e2nn.MultipleModule(
					self.feat_type_hid_scalar + self.feat_type_hid_vector,
					['relu']*len(self.feat_type_hid_scalar) + ['norm']*len(self.feat_type_hid_vector),
					[(relu, 'relu'), (norm_relu, 'norm')]
		)
		
		self.repr = nn.Sequential(
			e2nn.R2Conv(self.feat_type_in, 
						self.feat_type_hid_scalar + self.feat_type_hid_vector, 
						kernel_size=3, padding=1),
			nonlinearity_hid,

			e2nn.R2Conv(self.feat_type_hid_scalar + self.feat_type_hid_vector, 
						self.feat_type_hid_scalar + self.feat_type_hid_vector, 
						kernel_size=3, padding=1),
			nonlinearity_hid,
			
			e2nn.R2Conv(self.feat_type_hid_scalar + self.feat_type_hid_vector, 
						self.feat_type_out, 
						kernel_size=3, padding=1),
		)

	def forward(self, protein):
		x = e2nn.GeometricTensor(protein, self.feat_type_in)
		return self.repr(x)

class SpatialDock(nn.Module):
	def __init__(self, prot_field_size=50):
		super(SpatialDock, self).__init__()
		self.prot_field_size = prot_field_size
		self.convolution = EQProteinConv2D()

		self.get_attention = nn.Sequential(
			nn.Conv2d(9, 16, kernel_size=3, padding=1),
			nn.ReLU(),

			nn.Conv2d(16, 4, kernel_size=3, padding=1),
			nn.ReLU(),
			
			nn.Conv2d(4, 1, kernel_size=3, padding=1),	
			nn.Sigmoid()		
		)

	def forward(self, rep_rec, rep_lig):
		translations = self.convolution(rep_rec, rep_lig)
		attention = self.get_attention(translations.tensor)
		attention = torch.exp(attention)
		
		return attention
		

class AttentionDockModel(nn.Module):
	def __init__(self, num_angles=32, prot_field_size=50):
		super(AttentionDockModel, self).__init__()
		self.prot_field_size = prot_field_size
		self.num_angles = num_angles

		self.repr = EQRepresentation()
		self.dock = SpatialDock()

		translation = torch.zeros(num_angles, 2)
		angle = torch.from_numpy(np.linspace(0, 2*np.pi, self.num_angles)).to(dtype=torch.float32)
		mat_col1 = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros(num_angles)], dim=1)
		mat_col2 = torch.stack([-torch.sin(angle), torch.cos(angle), torch.zeros(num_angles)], dim=1)
		mat_col3 = torch.cat([translation, torch.ones(num_angles, 1)], dim=1)
		self.mat = torch.stack([mat_col1, mat_col2, mat_col3], dim=2)

		
		coords = torch.linspace(-self.prot_field_size, self.prot_field_size, 2*self.prot_field_size)
		x = coords.repeat(2*self.prot_field_size, 1)
		y = x.transpose(0,1)
		coord_features = torch.stack([x,y], dim=0).unsqueeze(dim=1).repeat(1, self.num_angles, 1, 1)
		rot_features = torch.stack([torch.cos(angle), torch.sin(angle)], dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
		rot_features = rot_features.repeat(1,1,2*self.prot_field_size, 2*self.prot_field_size)
		self.field = torch.cat([coord_features, rot_features], dim=0).unsqueeze(dim=0)

				
	def forward(self, receptor, ligand):	
		batch_size = receptor.size(0)
		num_features = receptor.size(1)
		box_size = receptor.size(2)
		
		receptor = receptor.unsqueeze(dim=1).repeat(1,self.num_angles,1,1,1)
		receptor = receptor.view(batch_size*self.num_angles, num_features, box_size, box_size)
		
		ligand = ligand.unsqueeze(dim=1).repeat(1,self.num_angles,1,1,1)
		ligand = ligand.view(batch_size*self.num_angles, num_features, box_size, box_size)
		
		with torch.no_grad():
			mat = self.mat.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1).view(batch_size*self.num_angles, 3, 3)
			self.grid = affine_grid(mat[:, :2, :3], receptor.size()).to(dtype=torch.float32, device=receptor.device)

		ligand_rot = grid_sample(ligand, self.grid, mode='nearest')
		
		rec_rep = self.repr(receptor)
		lig_rep = self.repr(ligand_rot)
				
		attention = self.dock(rec_rep, lig_rep)
		attention = attention.view(batch_size, self.num_angles, 2*box_size, 2*box_size)
		attention = attention / torch.sum(attention, dim=[1,2,3], keepdim=True)
		
		x = torch.sum((attention.unsqueeze(dim=1))*self.field.to(device=attention.device), dim=[2,3,4])
		
		translation = x[:,:2]
		rotation = x[:,2:]
		norm = torch.sqrt((rotation*rotation).sum(dim=1, keepdim=True)+1E-5)
		rotation = rotation/norm
		
		
		mat_col1 = torch.stack([rotation[:,0], rotation[:,1], torch.zeros(batch_size).to(device=receptor.device)], dim=1)
		mat_col2 = torch.stack([-rotation[:,1], rotation[:,0], torch.zeros(batch_size).to(device=receptor.device)], dim=1)
		mat_col3 = torch.cat([-translation, torch.ones(batch_size, 1).to(device=receptor.device)], dim=1)
		mat = torch.stack([mat_col1, mat_col2, mat_col3], dim=2)

		return mat[:, :2, :3]

if __name__=='__main__':
	def gaussian(tensor, center=(0,0), sigma=1.0):
		center = center[0] + tensor.size(0)/2, center[1] + tensor.size(1)/2
		for x in range(tensor.size(0)):
			for y in range(tensor.size(1)):
				r2 = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])
				arg = torch.tensor([-r2/sigma])
				tensor[x,y] = torch.exp(arg)

	a = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	b = torch.zeros(50, 50, dtype=torch.float, device='cpu')
	gaussian(a, center=(0,0), sigma=5)
	gaussian(b, center=(-10,-10), sigma=5)
	a = a.unsqueeze(dim=0).unsqueeze(dim=1)
	b = b.unsqueeze(dim=0).unsqueeze(dim=1)

	sdock = AttentionDockModel()
	sdock(a, b)
