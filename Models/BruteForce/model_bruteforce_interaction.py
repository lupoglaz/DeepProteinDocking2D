import torch
from torch import nn
import numpy as np
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter

class BruteForceInteraction(nn.Module):
    def __init__(self):
        super(BruteForceInteraction, self).__init__()

        self.softmax = torch.nn.Softmax(dim=0)
        self.dim = TorchDockingFilter().dim
        self.num_angles = TorchDockingFilter().num_angles

        self.threshold = 0
        self.F_0 = nn.Parameter(torch.ones(1) * self.threshold)

    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        # print(E.shape)
        deltaF = torch.logsumexp(-E, dim=(0,1,2)) - self.F_0
        pred_interact = -torch.sigmoid(deltaF)
        # pred_interact = -torch.div(1.0, (torch.exp(-deltaF) + 1.0)) + 1.0

        # print(pred_interact.shape)
        # U = torch.exp(-E)
        # deltaF = -torch.log(torch.mean(U))
        # pred_interact = -torch.div(1.0, (torch.exp(-deltaF + self.F_0) + 1.0)) + 1.0

        # ### unshifted free energy
        # U = torch.exp(-E)
        # deltaF = -torch.log(torch.mean(U))
        # pred_interact = -torch.div(1.0, (torch.exp(-deltaF + self.F_0) + 1.0)) + 1.0

        # ### new equation, unshifted free energy
        # U = torch.exp(-E)
        # deltaF = -torch.log(torch.sum(U)) - self.F_0
        # pred_interact = -torch.div(1.0, (torch.exp(-deltaF) + 1.0)) + 1.0

        # print(deltaF)
        print(self.F_0.item())
        # print(pred_interact)

        return pred_interact.squeeze()

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
