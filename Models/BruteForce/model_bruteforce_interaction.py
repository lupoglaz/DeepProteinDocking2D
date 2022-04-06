import torch
from torch import nn


class BruteForceInteraction(nn.Module):
    def __init__(self):
        super(BruteForceInteraction, self).__init__()
        # self.debug = False
        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.volume = torch.log(torch.tensor(100 ** 2))

    def forward(self, FFT_score, plotting=False, debug=False):
        ##TODO: pass BETA

        E = -FFT_score.squeeze()
        if len(E.shape) < 3:
            E = E.unsqueeze(0)
        if E.shape[0] > 1:
            self.volume = torch.log(E.shape[0]*torch.tensor(100 ** 2))
        # print(self.logdimsq)

        if E.shape[0] == 360:
            F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - self.volume)
        else:
            translationsF = torch.logsumexp(-E, dim=(1, 2))
            F = -(translationsF - self.volume)
            F = torch.mean(F, dim=0)
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
