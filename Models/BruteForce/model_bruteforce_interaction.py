import torch
from torch import nn


class BruteForceInteraction(nn.Module):
    def __init__(self):
        super(BruteForceInteraction, self).__init__()
        # self.debug = False
        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.logdimsq = torch.log(torch.tensor(100 ** 2))

    def forward(self, FFT_score, plotting=False, debug=False):
        E = -FFT_score
        if len(E.shape) > 3:
            E = E.squeeze()
        if len(E.shape) < 3:
            E = E.unsqueeze(0)
        if E.shape[0] > 1:
            # print(E.shape[0])
            self.logdimsq = torch.log(E.shape[0]*torch.tensor(100 ** 2))
        # print(self.logdimsq)

        # F = -torch.logsumexp(-E, dim=(0, 1, 2))
        F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.logdimsq)
        deltaF = F - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        if debug:
            with torch.no_grad():
                print('\n(F - F_0): ', deltaF.item())
                print('F_0: ', self.F_0.item())

        return pred_interact.squeeze(), deltaF.squeeze(), F, self.F_0


if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
    print(list(BruteForceInteraction().parameters()))
