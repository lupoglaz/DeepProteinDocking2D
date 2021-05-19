import torch
from torch import nn
from ProteinConvolution2D import ProteinConv2D
from ..Multiplication import ImageCrossMultiply

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
    convolve = ProteinConv2D()
    mult = ImageCrossMultiply()
    
    a = torch.zeros(50, 50, dtype=torch.float, device='cpu')
    b = torch.zeros(50, 50, dtype=torch.float, device='cpu')
    gaussian(a, (0,0), 1.0)
    gaussian(b, (10,0), 1.0)    
    a = a.unsqueeze(dim=0).unsqueeze(dim=1)
    b = b.unsqueeze(dim=0).unsqueeze(dim=1)

    d = pc(a, b).squeeze()

    T = torch.tensor([[-10, 0]], dtype=torch.float, device='cpu')
    m = mult(a, b, T)
    print(m, d[40, 50])
    
    
    f = plt.figure(figsize=(20,5))
    plt.subplot(1,3,1)
    plt.imshow(a.squeeze())
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(b.squeeze())
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.imshow(d)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

