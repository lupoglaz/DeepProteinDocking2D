import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pylab as plt
import random

class Energy(nn.Module):
    def __init__(self):
        super(Energy, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([0.5]))
        self.mu = nn.Parameter(torch.tensor([10.0]))
    
    def forward(self, x):
        return self.sigma*(x - self.mu)*(x - self.mu)

def energy_ans(x):
    return x*x

if __name__=='__main__':
    num_epochs = 300
    num_steps = 50
    Ntrain = 100
    mem_max = 20
    batch_size = 16

    x_train = np.random.randn(Ntrain)
    mem = [torch.rand(1) for i in range(batch_size)]
    
    energy = Energy()
    energy_optimizer = optim.Adam(energy.parameters(), lr=0.1)

    f = plt.figure(figsize=(12,6))
    ax = f.add_subplot(111)
    
    for i in range(num_epochs):
        x_batch = random.choices(mem, k=batch_size)
        x_batch = torch.stack(x_batch, dim=0)
        x_batch.requires_grad_()
        energy.eval()
        energy.mu.requires_grad = False
        energy.sigma.requires_grad = False
        langevin_optimizer = optim.Adam([x_batch], lr=0.1)
        x_traces = [[] for i in range(batch_size)]
        e_traces = [[] for i in range(batch_size)]
        for j in range(num_steps):
            langevin_optimizer.zero_grad()
            E = energy(x_batch)
            L = E.mean()
            L.backward()
            x_batch.grad += torch.randn(batch_size,1)*0.1
            langevin_optimizer.step()
            
            for k in range(batch_size):
                x_traces[k].append(x_batch[k].item())
                e_traces[k].append(E[k].item())
        
        x_batch = x_batch.detach()
        for j in range(batch_size):
            mem.append(x_batch[j].clone())
            if len(mem)>mem_max:
                mem.pop(0)
        
        plt.cla()
        for k in range(batch_size):
            ax.plot(x_traces[k], e_traces[k], 'r-.')
        ax.scatter(x_batch.numpy(), energy(x_batch).detach().numpy(), marker='x', c='g')
        ax.scatter(x_train, energy_ans(x_train), marker='o', c='b')
        plt.draw()
        plt.pause(.01)
        
        energy.train()
        energy.mu.requires_grad = True
        energy.sigma.requires_grad = True
        energy_optimizer.zero_grad()
        train_batch = torch.from_numpy(np.random.choice(x_train, batch_size))
        L = (energy(train_batch) - energy(x_batch)).mean()
        L.backward()
        energy_optimizer.step()
        print(L.item(), energy.mu.item(), energy.sigma.item())