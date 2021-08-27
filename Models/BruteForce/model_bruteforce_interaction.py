import torch
from torch import nn
import numpy as np

class BruteForceInteraction(nn.Module):
    def __init__(self):
        super(BruteForceInteraction, self).__init__()

        self.F_0 = nn.Parameter(torch.ones(1) * -25.0)

    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        ## normalize E?
        with torch.no_grad():
            norm, _ = torch.max(-E.view(1, 360*100*100), dim=1)
            norm = norm.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)

        shiftedE = -E - norm
        U = torch.exp(shiftedE)
        deltaF = -torch.log(torch.mean(U)) + norm.squeeze()
        pred_interact = -torch.div(1.0, (torch.exp(-deltaF + self.F_0) + 1.0)) + 1.0

        print(self.F_0.item())

        return pred_interact.squeeze()

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
