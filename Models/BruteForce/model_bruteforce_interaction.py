import torch
from torch import nn


class BruteForceInteraction(nn.Module):
    def __init__(self):
        super(BruteForceInteraction, self).__init__()

        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        deltaF = -torch.logsumexp(-E, dim=(0, 1, 2)) - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        with torch.no_grad():
            print('\n(deltaF - F_0): ', deltaF.item())
            print('F_0: ', self.F_0.item())

        return pred_interact.squeeze(), deltaF.squeeze()


if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
    print(list(BruteForceInteraction().parameters()))
