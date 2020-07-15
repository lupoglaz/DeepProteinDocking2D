import torch
from torch import nn
from convolution import EQProteinConv2D

import seaborn
from matplotlib import pylab as plt
from e2cnn import gspaces
from e2cnn import nn as e2nn
import numpy as np

def gaussian(tensor, center=(0,0), sigma=1.0):
    center = center[0] + tensor.size(0)/2, center[1] + tensor.size(1)/2
    for x in range(tensor.size(0)):
        for y in range(tensor.size(1)):
            r2 = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])
            arg = torch.tensor([-r2/sigma])
            tensor[x,y] = torch.exp(arg)
    
def vector_field(tensor, center=(0,0), sigma=1.0, direction=0.0):
    center = center[0] + tensor.size(1)/2, center[1] + tensor.size(2)/2
    for x in range(tensor.size(1)):
        for y in range(tensor.size(2)):
            r2 = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])
            arg = torch.tensor([-r2/sigma])
            tensor[1, x, y] = torch.exp(arg)*np.cos(direction)
            tensor[2, x, y] = torch.exp(arg)*np.sin(direction)
    
if __name__=='__main__':
    r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=5)
    feat_type_in = e2nn.FieldType(r2_act, [r2_act.trivial_repr, r2_act.irrep(1)])
    
    
    a = torch.zeros(3, 50, 50, dtype=torch.float, device='cpu')
    b = torch.zeros(3, 50, 50, dtype=torch.float, device='cpu')
    gaussian(a[0,:,:], (0,0), 1.0)
    vector_field(a, (0,0), 50.0, 0.0)
    gaussian(b[0,:,:], (10,0), 1.0)
    vector_field(b, (10,0), 50.0, 0.5*np.pi/2.0)

    a = e2nn.GeometricTensor(a.unsqueeze(dim=0), feat_type_in)
    b = e2nn.GeometricTensor(b.unsqueeze(dim=0), feat_type_in)

    pc = EQProteinConv2D()
    
    d = pc(a, b)
    d = d.tensor.squeeze()
    
    X, Y = np.meshgrid(range(50), range(50))
    X2, Y2 = np.meshgrid(range(100), range(100))
    
    f = plt.figure(figsize=(20,10))
    plt.subplot(2,4,1)
    plt.imshow(a.tensor[0,0,:,:])
    plt.colorbar()
    plt.subplot(2,4,5)
    plt.quiver(X, Y, a.tensor[0,1, ...], a.tensor[0,2, ...], units='xy')
    
    plt.subplot(2,4,2)
    plt.imshow(b.tensor[0,0,:,:])
    plt.colorbar()
    plt.subplot(2,4,6)
    plt.quiver(X, Y, b.tensor[0,1, ...], b.tensor[0,2, ...], units='xy')
    

    plt.subplot(2,4,3)
    plt.imshow(d[0,:,:])
    plt.colorbar()
    plt.subplot(2,4,7)
    plt.quiver(X2, Y2, d[1, ...], d[2, ...], units='xy')
    
    plt.subplot(2,4,4)
    plt.imshow(d[5,:,:])
    plt.colorbar()
    plt.subplot(2,4,8)
    plt.quiver(X2, Y2, d[3, ...], d[4, ...], units='xy')

    plt.tight_layout()
    plt.show()

