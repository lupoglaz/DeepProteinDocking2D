import torch
from torch import nn
from torch import optim
from convolution import ProteinConv2D, convolve

import seaborn
from matplotlib import pylab as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

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
    b = torch.zeros(50, 50, dtype=torch.float, device='cpu').requires_grad_()
    c = torch.zeros(100, 100, dtype=torch.float, device='cpu')
    gaussian(a, (0,0), 1.0)
    gaussian(c, (-10,0), 2.0)

    loss_data = []
    opt_data = []
    optimizer = optim.Adam([b], lr = 0.05)
    for i in range(100):
        optimizer.zero_grad()
        d = pc(a.unsqueeze(dim=0).unsqueeze(dim=1), b.unsqueeze(dim=0).unsqueeze(dim=1)).squeeze()
        loss = torch.sum((d-c)*(d-c))
        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            opt_data.append(b.detach().numpy())
            loss_data.append(loss.detach().item())
        
        
    f = plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.plot(loss_data)

    plt.subplot(1,3,2)
    plt.imshow(a.detach().numpy())
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(b.detach().numpy())
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    

