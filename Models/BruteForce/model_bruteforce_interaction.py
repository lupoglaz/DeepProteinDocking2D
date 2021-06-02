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
        self.FoI_weights = nn.Parameter(torch.rand(1, 360, 1, 1)).cuda()
        self.dim = TorchDockingFilter().dim
        self.num_angles = TorchDockingFilter().num_angles

        # self.scal = 1
        # self.vec = 7
        # self.SO2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=4)
        # self.feat_type_in1 = enn.FieldType(self.SO2, 1 * [self.SO2.trivial_repr])
        # self.feat_type_out1 = enn.FieldType(self.SO2, self.scal * [self.SO2.irreps['irrep_0']] + self.vec * [self.SO2.irreps['irrep_1']])
        # self.feat_type_out_final = enn.FieldType(self.SO2, 1 * [self.SO2.irreps['irrep_0']] + 1 * [self.SO2.irreps['irrep_1']])
        #
        # self.kernel = 5
        # self.pad = self.kernel//2
        # self.stride = 1
        # self.dilation = 1
        #
        # self.netSE2 = enn.SequentialModule(
        #     enn.R2Conv(self.feat_type_in1, self.feat_type_out1, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad , bias=False),
        #     enn.NormNonLinearity(self.feat_type_out1, function='n_relu', bias=False),
        #     enn.R2Conv(self.feat_type_out1, self.feat_type_out_final, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad, bias=False),
        #     enn.NormNonLinearity(self.feat_type_out_final, function='n_relu', bias=False),
        # )

        self.kernel = 5
        self.pad = self.kernel//2
        self.stride = 1
        self.dilation = 1
        self.conv3D = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=self.kernel, padding=self.pad, stride=self.stride, dilation=self.dilation, bias=False),
            nn.ReLU()
            # nn.Softplus()
        )


    def forward(self, FFT_score, plotting=False):
        softmax = torch.nn.Softmax2d()
        P = softmax(FFT_score.unsqueeze(0)).reshape(self.num_angles, self.dim, self.dim)

        B = self.conv3D(FFT_score.unsqueeze(0).unsqueeze(0))

        # E = -torch.log(torch.sum(P * B)) ## sum(P * B) == exp(-E) => -log(exp(-E)) = E
        # pred_interact = torch.exp(-E) / (torch.exp(-E) + 1)

        pred_interact = torch.sum(P * B) / (torch.sum(P * B) + 1)


        # if eval and plotting:
        #     with torch.no_grad():
        #         plt.close()
        #         plt.figure(figsize=(8, 8))
        #         if rec_feat.shape[-1] < receptor.shape[-1]:
        #             pad_size = (receptor.shape[-1] - rec_feat.shape[-1])//2
        #             if rec_feat.shape[-1] % 2 == 0:
        #                 rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        #                 lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        #             else:
        #                 rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
        #                                  value=0)
        #                 lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
        #                                value=0)
        #             # print('padded shape', rec_feat.shape)
        #         rec_plot = np.hstack((receptor.squeeze().detach().cpu(), rec_feat[0].squeeze().detach().cpu(),
        #                               rec_feat[1].squeeze().detach().cpu()))
        #         lig_plot = np.hstack((ligand.squeeze().detach().cpu(), lig_feat[0].squeeze().detach().cpu(),
        #                               lig_feat[1].squeeze().detach().cpu()))
        #         # plt.imshow(np.vstack((rec_plot, lig_plot)), vmin=0, vmax=1)
        #         plt.imshow(np.vstack((rec_plot, lig_plot)))
        #         plt.title('Input                   F1_bulk                    F2_bound')
        #         plt.colorbar()
        #         plt.savefig('figs/Feats_InteractionBruteForceTorchFFT_SE2Conv2D_++-Score_feats_'+str(torch.argmax(FFT_score))+'.png')
        #         plt.show()

        return pred_interact

if __name__ == '__main__':
    print('works')
    print(BruteForceInteraction())
