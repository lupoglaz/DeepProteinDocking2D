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

        # self.threshold = 0.0
        # self.F_0 = nn.Parameter(torch.ones(1) * self.threshold)
        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        deltaF = -torch.logsumexp(-E, dim=(0,1,2)) - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        print('\ndeltaF', deltaF.item())
        print('F_0', self.F_0.item())
        # print(pred_interact)

        return pred_interact.squeeze(), deltaF.squeeze()

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
