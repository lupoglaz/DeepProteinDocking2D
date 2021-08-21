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

        # self.dummy = nn.Parameter(None)
        # self.register_parameter('F_0', self.dummy)

        # self.F_0 = nn.Parameter(torch.empty(0))

        # self.dummy = nn.Parameter(None)
        # self.register_parameter('F_0', None)

        # self.register_buffer('F_0', None)

        # self.F_0 = nn.Parameter(torch.ones(1)*-25.0)
        self.F_0 = nn.Parameter(-torch.rand(1))

    def forward(self, FFT_score, plotting=False):
        E = -FFT_score
        # print(self.F_0)
        # print(self.F_0.nelement())
        ## to try
        # if self.F_0 is None:
        #     print("X"*100)
        #     U_0 = torch.mean(torch.exp(-E))
        #     self.F_0 = nn.Parameter(-torch.log(U_0)).cuda()
        #     # self.F_0 = -torch.log(torch.mean(torch.exp(-E)))
        #     print('One time initialization of F_0 * mean(E)')
        #     print(self.F_0)

        U = torch.exp(-E)
        deltaF = -torch.log(torch.mean(U))
        # print('deltaF')
        pred_interact = torch.div(torch.exp(-deltaF + self.F_0), ((torch.exp(-deltaF + self.F_0) + 1.0)))
        # print(pred_interact.item(), self.F_0.item())
        return pred_interact.squeeze()


        # Pflatsm = self.softmax(-E.flatten()).squeeze()
        # P = Pflatsm.reshape(self.num_angles, self.dim, self.dim)
        # print(P.shape, torch.sum(Pflatsm))

        ### eq 1.5
        # B = self.conv3D(E.unsqueeze(0).unsqueeze(0)).squeeze()
        # pred_interact = torch.sum(B * P) / (torch.sum(P))

        ### eq 10
        # B = self.conv3D(E.unsqueeze(0).unsqueeze(0)).squeeze()
        # eP = torch.sum(B * P) / (torch.sum((1-B)*P))
        # pred_interact = eP / (eP + 1) ## eq 7 substituted
        #
        # return pred_interact

        ## georgy code
        # with torch.no_grad():
        #     minE, _ = torch.min(E.view(1, self.num_angles*self.dim*self.dim), dim=1)
        #     minE = minE.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)


        # shifted_E = -E + minE #shifts all "scores" to be <= 0
        # U = torch.exp(shifted_E)
        # # print(torch.min(-E), torch.min(shifted_E))
        # pred_interact = -torch.log(torch.mean(U))
        # # print(pred_interact.item(), norm.squeeze().item(), self.F_0.item())
        # return pred_interact - self.F_0


        # minE = -torch.sum(-E * P)
        # threshold = torch.mean(-E) * -self.F_0
        # print(torch.mean(-E).item(), -self.F_0.item())
        # print(minE.item(), threshold.item())
        # return minE - threshold


        # avgE = -torch.sum(-E * P)
        # deltaF =
        # threshold = torch.mean(-E) * -self.F_0
        # print(torch.mean(-E).item(), -self.F_0.item())
        # print(minE.item(), threshold.item())
        # return minE - threshold


        # blah = torch.sum(P * torch.mean(torch.exp(E)))
        # U = P * -E
        # pred_interact = -torch.log(torch.mean(U))
        # print(pred_interact.item(), self.F_0.item())
        # return pred_interact - self.F_0

        # minE = torch.sum(E * P)
        # threshold = -torch.log(torch.sum(-E)) #- self.F_0
        # print(minE.item(), threshold.item())
        # return minE #- threshold

        # minE = -torch.sum(-E * P)
        #
        # interaction_threshold = self.F_0
        #
        # pred_interact = 1.0 - torch.exp(-minE * interaction_threshold)
        #
        # print(interaction_threshold.item())
        #
        # return pred_interact


        # with torch.no_grad():
        #     EP = -E * P
        #     maxEP, _ = torch.max(EP.view(1, self.num_angles*self.dim*self.dim), dim=1)
        #     maxEP = maxEP.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)


        # shiftedEP = EP #- maxEP #shifts all "scores" to be <= 0
        # U = torch.exp(shiftedEP)
        # U = shiftedEP
        # pred_interact = -torch.log(torch.mean(U))

        # sumPE = torch.sum(P*-E)
        # pred_interact = -(sumPE)
        #
        # # print(torch.max(shiftedEP).item(), torch.max(U).item()) # prints 0 for max of shifted down dist, and 1 for max of
        # # print(pred_interact.item(), self.F_0.item())
        # return pred_interact - (torch.mean(E))
        # # return pred_interact - (self.F_0 + -torch.log(torch.mean(torch.exp(-E))))


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
