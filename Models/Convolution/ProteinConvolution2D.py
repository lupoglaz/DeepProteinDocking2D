import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as e2nn
import math as m

def convolve(volume1, volume2, conj):
	cplx_rec = torch.fft.rfft2(volume1, dim=(-2, -1))
	cplx_lig = torch.fft.rfft2(volume2, dim=(-2, -1))

	if conj:
		return torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1))
	else:
		return torch.fft.irfft2(cplx_rec * cplx_lig, dim=(-2, -1))


# def convolve(volume1, volume2, conj):
# 	box_size = volume1.size(1)
#
# 	cv1 = torch.fft.rfft2(volume1, dim=(-2,-1))
# 	cv2 = torch.fft.rfft2(volume2, dim=(-2,-1))
# 	if conj:
# 		re = cv1[:,:,:].real * cv2[:,:,:].real + cv1[:,:,:].imag * cv2[:,:,:].imag
# 		im = -cv1[:,:,:].real * cv2[:,:,:].imag + cv1[:,:,:].imag * cv2[:,:,:].real
# 	else:
# 		re = cv1[:,:,:].real * cv2[:,:,:].real - cv1[:,:,:].imag * cv2[:,:,:].imag
# 		im = cv1[:,:,:].real * cv2[:,:,:].imag + cv1[:,:,:].imag * cv2[:,:,:].real
#
# 	cconv = torch.view_as_complex(torch.stack([re, im], dim=3))
#
# 	return torch.fft.irfft2(cconv,  dim=(-2,-1), s=(box_size, box_size))


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
	def __init__(self, full=True):
		super(ProteinConv2D, self).__init__()
		self.protConvolve = ProteinConv2DFunction()
		self.full = full
	
	def forward(self, volume1, volume2):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		
		if not self.full: #lower triangle product
			volume1_unpacked = []
			volume2_unpacked = []
			for i in range(0, num_features):
				volume1_unpacked.append(volume1[:,0:num_features-i,:,:])
				volume2_unpacked.append(volume2[:,i:num_features,:,:])
			volume1 = torch.cat(volume1_unpacked, dim=1)
			volume2 = torch.cat(volume2_unpacked, dim=1)
			num_output_features = volume1.size(1)
		else: #complete product
			volume1 = volume1.unsqueeze(dim=2).repeat(1, 1, num_features, 1, 1)
			volume2 = volume2.unsqueeze(dim=1).repeat(1, num_features, 1, 1, 1)
			num_output_features = num_features*num_features
		
		volume1 = volume1.view(batch_size*num_output_features, volume_size, volume_size)
		volume2 = volume2.view(batch_size*num_output_features, volume_size, volume_size)
		input_volume1 = F.pad(volume1, (0, volume_size, 0, volume_size)).contiguous()
		input_volume2 = F.pad(volume2, (0, volume_size, 0, volume_size)).contiguous()
		
		circ_volume = self.protConvolve.apply(input_volume1, input_volume2)
		output_volume_size = circ_volume.size(2)
		circ_volume = circ_volume.view(batch_size, num_output_features, output_volume_size, output_volume_size)

		return SwapQuadrants2DFunction.apply(circ_volume)
