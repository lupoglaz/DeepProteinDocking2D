import torch
from torch import nn
from convolution import ProteinConv2D, convolve

import seaborn
from matplotlib import pylab as plt


def gaussian(tensor, center=(0,0), sigma=1.0):
    center = center[0] + tensor.size(0)/2, center[1] + tensor.size(1)/2
    for x in range(tensor.size(0)):
        for y in range(tensor.size(1)):
            r2 = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1])
            arg = torch.tensor([-r2/sigma])
            tensor[x,y] = torch.exp(arg)

if __name__=='__main__':
    pc = ProteinConv2D()
    a = torch.zeros(50, 50, dtype=torch.float, device='cpu')
    b = torch.zeros(50, 50, dtype=torch.float, device='cpu')
    gaussian(a, (0,0), 1.0)
    gaussian(b, (10,0), 1.0)
    c = convolve(a.unsqueeze(dim=0), b.unsqueeze(dim=0), True).squeeze()
    d = pc(a.unsqueeze(dim=0).unsqueeze(dim=1), b.unsqueeze(dim=0).unsqueeze(dim=1)).squeeze()
    
    f = plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    plt.imshow(a)
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.imshow(b)
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(c)
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.imshow(d)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

