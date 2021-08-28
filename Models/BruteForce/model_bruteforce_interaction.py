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

        # normalize E? ### NO GRADIENT
        # with torch.no_grad():
        #     norm, _ = torch.max(-E.view(1, 360*100*100), dim=1)
        #     norm = norm.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)

        # Pflatsm = self.softmax(-E.flatten()).squeeze()
        # P = Pflatsm.reshape(self.num_angles, self.dim, self.dim)
        # norm = torch.sum(P * -E)

        # print(norm)
        # shiftedE = -E - norm
        # E = -shiftedE
        # U = torch.exp(-E) + norm.squeeze()
        # deltaF = -torch.log(torch.mean(U))
        # pred_interact = -torch.div(1.0, (torch.exp(-deltaF + self.F_0) + 1.0)) + 1.0

        U = torch.exp(-E)
        deltaF = -torch.log(torch.mean(U))
        pred_interact = -torch.div(1.0, (torch.exp(-deltaF + self.F_0) + 1.0)) + 1.0

        # print(deltaF)
        print(self.F_0.item())
        # print(pred_interact)

        return pred_interact.squeeze()

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
