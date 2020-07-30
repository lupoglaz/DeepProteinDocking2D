import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as e2nn

def convolve(volume1, volume2, conj):
	batch_size = volume1.size(0)
	box_size = volume1.size(1)
	full_vol = box_size*box_size
	output = torch.zeros(batch_size, box_size, box_size, device=volume1.device, dtype=volume1.dtype)
	
	cv1 = torch.rfft(volume1, 2)
	cv2 = torch.rfft(volume2, 2)
	
	if conj:
		re = cv1[:,:,:,0]*cv2[:,:,:,0] + cv1[:,:,:,1]*cv2[:,:,:,1]
		im = -cv1[:,:,:,0]*cv2[:,:,:,1] + cv1[:,:,:,1]*cv2[:,:,:,0]
	else:
		re = cv1[:,:,:,0]*cv2[:,:,:,0] - cv1[:,:,:,1]*cv2[:,:,:,1]
		im = cv1[:,:,:,0]*cv2[:,:,:,1] + cv1[:,:,:,1]*cv2[:,:,:,0]
	
	cconv = torch.stack([re, im], dim=3)
	
	return torch.irfft(cconv, 2, signal_sizes=(box_size, box_size))

class ProteinConv2DFunction(Function):
	@staticmethod
	def forward(ctx, volume1, volume2):
		ctx.save_for_backward(volume1, volume2)
		return convolve(volume1, volume2, conj=True)

	@staticmethod
	def backward(ctx, grad_output):
		volume1, volume2 = ctx.saved_tensors
		gradVolume1 = convolve(grad_output, volume2, conj=False)
		gradVolume2 = convolve(volume1, grad_output, conj=True)
		return gradVolume1, gradVolume2

class SwapQuadrants2DFunction(Function):
	@staticmethod
	def forward(ctx, input_volume):
		batch_size = input_volume.size(0)
		num_features = input_volume.size(1)
		L = input_volume.size(2)
		L2 = int(L/2)
		output_volume = torch.zeros(batch_size, num_features, L, L, device=input_volume.device, dtype=input_volume.dtype)
		
		output_volume[:, :, :L2, :L2] = input_volume[:, :, L2:L, L2:L]
		output_volume[:, :, L2:L, L2:L] = input_volume[:, :, :L2, :L2]

		output_volume[:, :, L2:L, :L2] = input_volume[:, :, :L2, L2:L]
		output_volume[:, :, :L2, L2:L] = input_volume[:, :, L2:L, :L2]

		output_volume[:, :, L2:L, L2:L] = input_volume[:, :, :L2, :L2]
		output_volume[:, :, :L2, :L2] = input_volume[:, :, L2:L, L2:L]

		return output_volume

	@staticmethod
	def backward(ctx, grad_output):
		batch_size = grad_output.size(0)
		num_features = grad_output.size(1)
		L = grad_output.size(2)
		L2 = int(L/2)
		grad_input = torch.zeros(batch_size, num_features, L, L, device=grad_output.device, dtype=grad_output.dtype)

		grad_input[:, :, :L2, :L2] = grad_output[:, :, L2:L, L2:L]
		grad_input[:, :, L2:L, L2:L] = grad_output[:, :, :L2, :L2]

		grad_input[:, :, L2:L, :L2] = grad_output[:, :, :L2, L2:L]
		grad_input[:, :, :L2, L2:L] = grad_output[:, :, L2:L, :L2]

		grad_input[:, :, L2:L, L2:L] = grad_output[:, :, :L2, :L2]
		grad_input[:, :, :L2, :L2] = grad_output[:, :, L2:L, L2:L]

		return grad_input


class ProteinConv2D(nn.Module):
	def __init__(self, append_coords=False):
		super(ProteinConv2D, self).__init__()
		self.protConvolve = ProteinConv2DFunction()
	
	def forward(self, volume1, volume2):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		
		# volume1 = volume1.unsqueeze(dim=2).repeat(1, 1, num_features, 1, 1)
		# volume2 = volume2.unsqueeze(dim=1).repeat(1, num_features, 1, 1, 1)
		# volume1 = volume1.view(batch_size*num_features*num_features, volume_size, volume_size)
		# volume2 = volume2.view(batch_size*num_features*num_features, volume_size, volume_size)
		
		# input_volume1 = F.pad(volume1, (0, volume_size, 0, volume_size)).contiguous()
		# input_volume2 = F.pad(volume2, (0, volume_size, 0, volume_size)).contiguous()


		volume1_unpacked = []
		volume2_unpacked = []
		for i in range(0, num_features):
			volume1_unpacked.append(volume1[:,0:num_features-i,:,:])
			volume2_unpacked.append(volume2[:,i:num_features,:,:])
		volume1 = torch.cat(volume1_unpacked, dim=1)
		volume2 = torch.cat(volume2_unpacked, dim=1)
		
		num_output_features = volume1.size(1)
		volume1 = volume1.view(batch_size*num_output_features, volume_size, volume_size)
		volume2 = volume2.view(batch_size*num_output_features, volume_size, volume_size)
		input_volume1 = F.pad(volume1, (0, volume_size, 0, volume_size)).contiguous()
		input_volume2 = F.pad(volume2, (0, volume_size, 0, volume_size)).contiguous()
		
		circ_volume = self.protConvolve.apply(input_volume1, input_volume2)
				
		# num_output_features = num_features*num_features
		output_volume_size = circ_volume.size(2)

		circ_volume = circ_volume.view(batch_size, num_output_features, output_volume_size, output_volume_size)

		return SwapQuadrants2DFunction.apply(circ_volume)

class EQProteinConv2D(nn.Module):
	def __init__(self, append_coords=False):
		super(EQProteinConv2D, self).__init__()
		self.protConvolve = ProteinConv2DFunction()
	
	def forward(self, volume1, volume2):
		assert volume1.tensor.size(0) == volume2.tensor.size(0)
		batch_size = volume1.tensor.size(0)

		assert volume1.tensor.size(1) == volume2.tensor.size(1)
		num_features = volume1.tensor.size(1)

		assert volume1.tensor.size(2) == volume2.tensor.size(2)
		assert volume1.tensor.size(3) == volume2.tensor.size(3)
		assert volume1.tensor.size(2) == volume1.tensor.size(3)
		volume_size = volume1.tensor.size(2)

		volume1_unfolded = []
		volume2_unfolded = []
				
		index_output = 0
		index_volume1 = 0
		to_reduce = []
		output_representations = []
		output_gspace = volume1.type.gspace
		for rep1 in volume1.type.representations:
			index_volume2 = 0
			for rep2 in volume2.type.representations:
				# print(rep1, rep2)
				if not (rep1.irreducible and rep2.irreducible):
					raise Exception("Reducible representation", str(rep1), str(rep2))
				
				if rep1.size == 1 and rep2.size == 1:
					volume1_unfolded.append(volume1.tensor[:, index_volume1:index_volume1+rep1.size, :, :])
					volume2_unfolded.append(volume2.tensor[:, index_volume2:index_volume2+rep2.size, :, :])
					output_representations.append(output_gspace.trivial_repr)
				
				elif rep1.size == 2 and rep2.size == 1:
					volume1_unfolded.append(volume1.tensor[:, index_volume1:index_volume1+rep1.size, :, :])
					
					volume2_unfolded.append(volume2.tensor[:, index_volume2:index_volume2+rep2.size, :, :])
					volume2_unfolded.append(volume2.tensor[:, index_volume2:index_volume2+rep2.size, :, :])
					
					output_representations.append(output_gspace.irrep(1))
				
				elif rep1.size == 1 and rep2.size == 2:
					volume1_unfolded.append(volume1.tensor[:, index_volume1:index_volume1+rep1.size, :, :])
					volume1_unfolded.append(volume1.tensor[:, index_volume1:index_volume1+rep1.size, :, :])
					
					volume2_unfolded.append(volume2.tensor[:, index_volume2:index_volume2+rep2.size, :, :])
					
					output_representations.append(output_gspace.irrep(1))
				
				elif rep1.size == 2 and rep2.size == 2:
					volume1_unfolded.append(volume1.tensor[:, index_volume1:index_volume1+rep1.size, :, :])
					volume2_unfolded.append(volume2.tensor[:, index_volume2:index_volume2+rep2.size, :, :])
					
					to_reduce.append( (index_output, index_output+max(rep1.size, rep2.size)) )
					
					output_representations.append(output_gspace.trivial_repr)
				else:
					raise Exception("Representations higher than dim2 not supported:", str(rep1), str(rep2))
				
				index_volume2 += rep2.size
				index_output += max(rep1.size, rep2.size)

			index_volume1 += rep1.size
		
		output_fields = e2nn.FieldType(output_gspace, output_representations)

		tmp_volume1 = torch.cat(volume1_unfolded, dim=1)
		tmp_volume2 = torch.cat(volume2_unfolded, dim=1)
		# print(tmp_volume1.size(), tmp_volume2.size())

		tmp_volume1 = tmp_volume1.view(batch_size*tmp_volume1.size(1), volume_size, volume_size)
		tmp_volume2 = tmp_volume2.view(batch_size*tmp_volume2.size(1), volume_size, volume_size)
		# print(tmp_volume1.size(), tmp_volume2.size())

		tmp_volume1 = F.pad(tmp_volume1, (0, volume_size, 0, volume_size)).contiguous()
		tmp_volume2 = F.pad(tmp_volume2, (0, volume_size, 0, volume_size)).contiguous()
		# print(tmp_volume1.size(), tmp_volume2.size())
		
		circ_volume = self.protConvolve.apply(tmp_volume1, tmp_volume2)		
		output_volume_size = circ_volume.size(2)
		circ_volume = circ_volume.view(batch_size, index_output, output_volume_size, output_volume_size)

		if len(to_reduce) == 0:
			reduced_circ_volume = circ_volume
		else:
			reduced_circ_volume = [circ_volume[:, :to_reduce[0][0], :, :]]
			for ind_start, ind_end in to_reduce:
				reduced_circ_volume += [torch.sum(circ_volume[:, ind_start:ind_end, :, :], dim=1, keepdim=True)]
			reduced_circ_volume += [circ_volume[:, to_reduce[-1][1]:, :, :]]
			reduced_circ_volume = torch.cat(reduced_circ_volume, dim=1)

		swapped_volume = SwapQuadrants2DFunction.apply(reduced_circ_volume)
		return e2nn.GeometricTensor(swapped_volume, output_fields)
		