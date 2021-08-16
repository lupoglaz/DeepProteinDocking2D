import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFilter import TorchDockingFilter

class BruteForceInteraction(nn.Module):

    def __init__(self):
        super(BruteForceInteraction, self).__init__()
        self.softmax = torch.nn.Softmax(dim=0)

        self.dim = TorchDockingFilter().dim
        self.num_angles = TorchDockingFilter().num_angles

        self.T = nn.Parameter(torch.rand(1))
        self.F_0 = nn.Parameter(torch.rand(1))
        #
        # self.kernel = 5
        # self.pad = self.kernel//2
        # self.stride = 1
        # self.dilation = 1
        # self.conv3D = nn.Sequential(
        #     nn.Conv3d(1, 4, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=True),
        #     nn.ReLU(),
        #     nn.Conv3d(4, 1, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=True),
        #     nn.Sigmoid(),
        # )


    def forward(self, FFT_score, plotting=False):
        E = -FFT_score

        Pflatsm = self.softmax(-E.flatten()).squeeze()
        P = Pflatsm.reshape(self.num_angles, self.dim, self.dim)
        # print(P.shape, torch.sum(Pflatsm))

        ### eq 1.5
        # B = self.conv3D(E.unsqueeze(0).unsqueeze(0)).squeeze()
        # pred_interact = torch.sum(B * P) / (torch.sum(P))

        # ### eq 10
        # B = self.conv3D(E.unsqueeze(0).unsqueeze(0)).squeeze()
        # eP = torch.sum(B * P) / (torch.sum((1-B)*P))
        # pred_interact = eP / (eP + 1) ## eq 7 substituted

        ### eq16
        # U = torch.exp(-E)
        # U = P * E
        # # U = P
        # print(torch.sum(U))
        # pred_interact = 1.0 + torch.log(torch.mean(U))
        # # pred_interact = -self.T * torch.log(torch.mean(U))
        # return -torch.sigmoid(pred_interact) #- self.F_0

        # U = P * E
        # U = torch.mean(P) * U
        # print(torch.sum(U))


        # U = torch.sum((P) * torch.sum(torch.exp(-E)))
        # U = torch.mean((P) * torch.sum(torch.exp(-E)))
        # pred_interact = -self.T * torch.log(U)

        # U = torch.sum(P * torch.sum(E))
        # U = torch.mean(P) * torch.mean(E)
        # print(torch.mean(P))
        U = torch.sum(torch.exp(-E))/(self.num_angles*self.dim**2)
        print(U)
        pred_interact = -self.T * torch.log(U)
        # pred_interact = -self.T * U
        return pred_interact #- self.F_0

        # print(expU)
        # print(integral)


        # if eval and plotting:
        #     with torch.no_grad():
        #         plt.close()
        #         plt.figure(figsize=(8, 8))
        #         minind = torch.argmin(E)
        #         maxind = torch.argmax(FFT_score)
        #         print('E min', torch.min(E), 'Score max', torch.max(FFT_score))
        #         plot_minindex = int(((minind / self.dim ** 2) * np.pi / 180.0) - np.pi)
        #         plot_maxindex = int(((maxind / self.dim ** 2) * np.pi / 180.0) - np.pi)
        #         plotScore = FFT_score.squeeze()[plot_maxindex, :, :].detach().cpu()
        #         plotE = E.squeeze()[plot_minindex, :, :].detach().cpu()
        #         plotP = P.squeeze()[plot_minindex, :, :].detach().cpu()
        #         plotB = B.squeeze()[plot_minindex, :, :].detach().cpu()
        #
        #         plot = np.hstack((plotScore, plotE, plotP, plotB))
        #         # plot = plotB
        #         plt.imshow(plot, vmin=int(torch.min(E)), vmax=int(torch.max(FFT_score)))
        #         plt.title('FFTscore'+str(int(torch.max(FFT_score).item()))+', Energy'+str(int(torch.min(E).item()))+', Probability, B indicator')
        #         plt.colorbar()
        #         plt.savefig('figs/Bmap_statphys_InteractionBruteForce'+str(minind.item())+'.png')
        #         plt.show()

        # return pred_interact - self.F_0
        # return deltaF

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
