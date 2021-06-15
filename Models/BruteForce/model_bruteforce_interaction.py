import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter
from e2cnn import nn as enn
from e2cnn import gspaces

class BruteForceInteraction(nn.Module):

    def __init__(self):
        super(BruteForceInteraction, self).__init__()
        self.softmax = torch.nn.Softmax2d()

        # self.FoI_weights = nn.Parameter(torch.rand(1, 360, 1, 1)).cuda()
        self.dim = TorchDockingFilter().dim
        self.num_angles = TorchDockingFilter().num_angles
        # self.softmin = torch.nn.Softmin(dim=1)

        self.kernel = 5
        self.pad = self.kernel//2
        self.stride = 1
        self.dilation = 1
        self.conv3D = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=False),
            nn.ReLU(),
            nn.Conv3d(4, 4, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=False),
            nn.ReLU(),
            nn.Conv3d(4, 1, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=False),
            nn.ReLU(),
        )


    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        P = self.softmax(-E.unsqueeze(0)).squeeze()
        # print(P.shape)

        # B = self.conv3D(-E.unsqueeze(0).unsqueeze(0)).squeeze()
        # B = self.conv3D(E.unsqueeze(0).unsqueeze(0)).squeeze()
        # print(B.shape)

        Esig = torch.sigmoid(-E)
        B = self.conv3D(Esig.unsqueeze(0).unsqueeze(0)).squeeze()


        pred_interact = torch.sum(P * B) / (torch.sum(P * B) + 1)

        # E = -torch.log(torch.sum(P * B)) ## sum(P * B) == exp(-E) => -log(exp(-E)) = E
        # pred_interact = torch.exp(-E) / (torch.exp(-E) + 1)

        if eval and plotting:
            with torch.no_grad():
                plt.close()
                plt.figure(figsize=(8, 8))
                minind = torch.argmin(E)
                plot_index = int(((minind / self.dim ** 2) * np.pi / 180.0) - np.pi)
                plotE = E.squeeze()[plot_index, :, :].detach().cpu()
                plotP = P.squeeze()[plot_index, :, :].detach().cpu()
                plotB = B.squeeze()[plot_index, :, :].detach().cpu()

                # plot = np.hstack((plotE, plotP, plotB, plotP*plotB))
                plot = plotB
                plt.imshow(plot)#, vmin=-1, vmax=1)
                plt.title('E map,       P map,       B map,        P*B')
                plt.colorbar()
                plt.savefig('figs/maxE_Emaps_Pmaps_Bmaps_statphys_InteractionBruteForce.png')
                plt.show()

        return pred_interact

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
