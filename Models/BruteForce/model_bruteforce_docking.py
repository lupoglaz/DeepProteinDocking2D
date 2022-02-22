import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from e2cnn import nn as enn
from e2cnn import gspaces


class BruteForceDocking(nn.Module):

    def __init__(self):
        super(BruteForceDocking, self).__init__()
        self.boundW = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crosstermW1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crosstermW2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.bulkW = nn.Parameter(torch.ones(1, requires_grad=True))

        self.scal = 1
        self.vec = 4

        self.SO2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=4)
        self.feat_type_in1 = enn.FieldType(self.SO2, 1 * [self.SO2.trivial_repr])
        self.feat_type_out1 = enn.FieldType(self.SO2, self.scal * [self.SO2.irreps['irrep_0']] + self.vec * [self.SO2.irreps['irrep_1']])
        self.feat_type_out_final = enn.FieldType(self.SO2, 1 * [self.SO2.irreps['irrep_0']] + 1 * [self.SO2.irreps['irrep_1']])

        self.kernel = 5
        self.pad = self.kernel//2
        self.stride = 1
        self.dilation = 1

        self.netSE2 = enn.SequentialModule(
            enn.R2Conv(self.feat_type_in1, self.feat_type_out1, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad, bias=False),
            enn.NormNonLinearity(self.feat_type_out1, function='n_relu', bias=False),
            enn.R2Conv(self.feat_type_out1, self.feat_type_out_final, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation, padding=self.pad, bias=False),
            enn.NormNonLinearity(self.feat_type_out_final, function='n_relu', bias=False),
            enn.NormPool(self.feat_type_out_final),
        )

    def forward(self, receptor, ligand, plotting=False):

        receptor_geomT = enn.GeometricTensor(receptor.unsqueeze(0), self.feat_type_in1)
        ligand_geomT = enn.GeometricTensor(ligand.unsqueeze(0), self.feat_type_in1)

        rec_feat = self.netSE2(receptor_geomT).tensor.squeeze()
        lig_feat = self.netSE2(ligand_geomT).tensor.squeeze()

        FFT_score = TorchDockingFFT(dim=rec_feat.shape[-1]*2, num_angles=360).dock_global(
            rec_feat,
            lig_feat,
            weight_bound=self.boundW,
            weight_crossterm1=self.crosstermW1,
            weight_crossterm2=self.crosstermW2,
            weight_bulk=self.bulkW
        )

        #### Plot shape features
        if plotting and eval:
            with torch.no_grad():
                self.plot_feats(rec_feat, lig_feat, receptor, ligand, FFT_score)

        return FFT_score

    def plot_feats(self, rec_feat, lig_feat, receptor, ligand, FFT_score):
        print('\nLearned scoring coefficients')
        print('bound', str(self.boundW.item())[:6])
        print('crossterm1', str(self.crosstermW1.item())[:6])
        print('crossterm2', str(self.crosstermW2.item())[:6])
        print('bulk', str(self.bulkW.item())[:6])
        plt.close()
        plt.figure(figsize=(8, 8))
        if rec_feat.shape[-1] < receptor.shape[-1]:
            pad_size = (receptor.shape[-1] - rec_feat.shape[-1]) // 2
            if rec_feat.shape[-1] % 2 == 0:
                rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
                lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            else:
                rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
                                 value=0)
                lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
                                 value=0)

        pad_size = (receptor.shape[-1]) // 2
        receptor = F.pad(receptor, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        ligand = F.pad(ligand, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        rec_plot = np.hstack((receptor.squeeze().t().detach().cpu(),
                              rec_feat[0].squeeze().t().detach().cpu(),
                              rec_feat[1].squeeze().t().detach().cpu()))
        lig_plot = np.hstack((ligand.squeeze().t().detach().cpu(),
                              lig_feat[0].squeeze().t().detach().cpu(),
                              lig_feat[1].squeeze().t().detach().cpu()))

        plt.imshow(np.vstack((rec_plot, lig_plot)), vmin=0, vmax=1)  # plot scale limits
        # plt.imshow(np.vstack((rec_plot, lig_plot)))
        # plt.title('Input'+' '*33+'F1_bulk'+' '*33+'F2_bound')
        plt.title('Input', loc='left')
        plt.title('F1_bulk')
        plt.title('F2_bound', loc='right')
        # plt.colorbar()
        plt.grid(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom
        plt.savefig('figs/docking_feats_MinEnergy' + str(torch.argmin(-FFT_score).item())[:4] + '.png')
        # plt.show()

if __name__ == '__main__':
    print('works')
    print(BruteForceDocking())
    print(list(BruteForceDocking().parameters()))
