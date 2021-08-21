import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

class BruteForceInteraction(nn.Module):

    def __init__(self):
        super(BruteForceInteraction, self).__init__()

        self.F_0 = nn.Parameter(torch.ones(1)*-25.0)

    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        U = torch.exp(-E)
        deltaF = -torch.log(torch.mean(U))
        # pred_interact = torch.div(torch.exp(-deltaF + self.F_0), ((torch.exp(-deltaF + self.F_0) + 1.0)))
        pred_interact = -torch.div(1.0, ((torch.exp(-deltaF + self.F_0) + 1.0))) + 1.0

        print(self.F_0.item())

        return pred_interact.squeeze()

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
