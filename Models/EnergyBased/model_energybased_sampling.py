import torch
from torch import optim
import torch.nn as nn
import numpy as np
from matplotlib import pylab as plt

from plot_EBM import EBMPlotter
from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking
import random

import os
import sys
sys.path.append('/home/sb1638/')
from torch.autograd import Function
import torch.nn.functional as F
from e2cnn import nn as enn
from e2cnn import gspaces

# class ImageCrossMultiply(nn.Module):
#     def __init__(self, full=True):
#         super(ImageCrossMultiply, self).__init__()
#         self.full = full
#
#     def forward(self, volume1, volume2, alpha, dr):
#         batch_size = volume1.size(0)
#         num_features = volume1.size(1)
#         volume_size = volume1.size(2)
#         mults = []
#         perm = torch.tensor([1, 0], dtype=torch.long, device=volume1.device)
#         dr = -2.0 * dr[:, perm] / volume_size
#         T0 = torch.cat([torch.cos(alpha), -torch.sin(alpha)], dim=1)
#         T1 = torch.cat([torch.sin(alpha), torch.cos(alpha)], dim=1)
#         t = torch.stack([(T0 * dr).sum(dim=1), (T1 * dr).sum(dim=1)], dim=1)
#         T01 = torch.stack([T0, T1], dim=1)
#         A = torch.cat([T01, t.unsqueeze(dim=2)], dim=2)
#
#         grid = nn.functional.affine_grid(A, size=volume2.size())
#         volume2 = nn.functional.grid_sample(volume2, grid)
#         if not self.full:
#             volume1_unpacked = []
#             volume2_unpacked = []
#             for i in range(0, num_features):
#                 volume1_unpacked.append(volume1[:, 0:num_features - i, :, :])
#                 volume2_unpacked.append(volume2[:, i:num_features, :, :])
#             volume1 = torch.cat(volume1_unpacked, dim=1)
#             volume2 = torch.cat(volume2_unpacked, dim=1)
#         else:
#             volume1 = volume1.unsqueeze(dim=2).repeat(1, 1, num_features, 1, 1)
#             volume2 = volume2.unsqueeze(dim=1).repeat(1, num_features, 1, 1, 1)
#             volume1 = volume1.view(batch_size, num_features * num_features, volume_size, volume_size)
#             volume2 = volume2.view(batch_size, num_features * num_features, volume_size, volume_size)
#
#         mults = (volume1 * volume2).sum(dim=3).sum(dim=2)
#
#         return mults, volume2, grid
#
# class EQScoringModel(nn.Module):
#     def __init__(self, repr, num_features=1, prot_field_size=50):
#         super(EQScoringModel, self).__init__()
#         self.prot_field_size = prot_field_size
#
#         self.mult = ImageCrossMultiply()
#         self.repr = repr
#         self.SO2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=4)
#         self.feat_type_in1 = enn.FieldType(self.SO2, 1 * [self.SO2.trivial_repr])
#         # self.boundW = nn.Parameter(torch.ones(1, requires_grad=True))
#         # self.crosstermW1 = nn.Parameter(torch.ones(1, requires_grad=True))
#         # self.crosstermW2 = nn.Parameter(torch.ones(1, requires_grad=True))
#         # self.bulkW = nn.Parameter(torch.ones(1, requires_grad=True))
#
#         self.scorer = nn.Sequential(
#             nn.Linear(4, 1, bias=False)
#         )
#     #     with torch.no_grad():
#     #         self.scorer.apply(init_weights)
#     # #
#     # def scorer(self, pos_repr):
#     #     print(pos_repr.shape)
#         # return self.bulkW * pos_repr[0,0] + self.crosstermW1 * pos_repr[0,1] + self.crosstermW2 * pos_repr[0,2] - self.boundW * pos_repr[0,3]
#
#     def forward(self, receptor, ligand, alpha, dr):
#         receptor_geomT = enn.GeometricTensor(receptor.unsqueeze(0), self.feat_type_in1)
#         ligand_geomT = enn.GeometricTensor(ligand.unsqueeze(0), self.feat_type_in1)
#         rec_feat = self.repr(receptor_geomT).tensor
#         lig_feat = self.repr(ligand_geomT).tensor
#
#         pos_repr, _, A = self.mult(rec_feat, lig_feat, alpha, dr)
#
#         score = self.scorer(pos_repr)
#         # print(score.shape)
#         return score


class DockerEBM(nn.Module):
    def __init__(self):
        super(DockerEBM, self).__init__()
        self.docker = BruteForceDocking(dim=100, num_angles=1)
        self.dockingFFT = TorchDockingFFT(num_angles=1, angle=None, swap_plot_quadrants=False)

    def forward(self, receptor, ligand, rotation, plot_count=1, stream_name='trainset'):
        if 'trainset' not in stream_name:
            training = False
        else: training = True
        FFT_score = self.docker.forward(receptor, ligand, angle=rotation, plotting=True, training=training, plot_count=plot_count, stream_name=stream_name)
        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
            best_score = FFT_score[pred_txy[0], pred_txy[1]]
        return -best_score, pred_txy, FFT_score


class EnergyBasedModel(nn.Module):
    def __init__(self, device='cuda', num_samples=1, weight=1.0, step_size=1, sample_steps=1, experiment=None):
        super(EnergyBasedModel, self).__init__()
        self.debug = False
        # self.training = False
        # self.BF_init = False
        # self.wReg = True
        self.Force_reg = False
        if self.Force_reg:
            self.k = 1e-3
            self.eps = 1e-7

        # self.repr = BruteForceDocking().netSE2
        # self.EBMdocker = EQScoringModel(repr=self.repr)
        self.EBMdocker = DockerEBM()

        # self.buffer = SampleBuffer(num_buf_samples)
        # self.buffer2 = SampleBuffer(num_buf_samples)

        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.weight = weight
        self.step_size = step_size
        self.device = device

        self.plot_idx = 0

        self.experiment = experiment
        # self.path_IP = 'figs/IP_figs/' + self.experiment
        # self.path_FI = 'figs/FI_figs/' + self.experiment
        # self.path_LD = 'figs/FI_figs/' + self.experiment + '/LD_steps'

        try:
            if 'IP' in self.experiment:
                os.mkdir(self.path_IP)
            if 'FI' in self.experiment:
                os.mkdir(self.path_FI)
                os.mkdir(self.path_LD)
        except:
            print('dir already exists')

    def forward(self, neg_alpha, neg_dr, receptor, ligand, temperature='cold', plot_count=1, stream_name='trainset'):
        noise_alpha = torch.zeros_like(neg_alpha)
        noise_dr = torch.zeros_like(neg_dr)

        # self.requires_grad(False)
        self.EBMdocker.eval()

        neg_alpha.requires_grad_()
        neg_dr.requires_grad_()
        langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

        if temperature == 'cold':
            self.sig_dr = 0.05
            self.sig_alpha = 0.5
            # self.sig_dr = 5
            # self.sig_alpha = 5
            # self.sig_dr = 25
            # self.sig_alpha = 50

        if temperature == 'hot':
            self.sig_dr = 0.5
            self.sig_alpha = 5

        for i in range(self.sample_steps):
            if self.debug:
                if i == 0 or i == 1:
                    print('Before RandomForce LD', i)
                    print(neg_alpha)
                    print(neg_dr)

            langevin_opt.zero_grad()

            best_score, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name)
            # best_score = self.EBMdocker(receptor, ligand, neg_alpha, neg_dr)
            best_score.mean().backward()
            langevin_opt.step()

            # rand_dr = torch.normal(0, self.sig_dr, size=neg_dr.shape).cuda()
            # neg_dr = neg_dr + rand_dr
            # # neg_dr = neg_dr.clamp(-rec_feat.size(2), rec_feat.size(2))
            # rand_alpha = torch.normal(0, self.sig_alpha, size=neg_alpha.shape).cuda()
            # neg_alpha = neg_alpha + rand_alpha

            neg_dr = neg_dr + noise_dr.normal_(0, self.sig_dr)
            neg_alpha = neg_alpha + noise_alpha.normal_(0, self.sig_alpha)
            clamp_offset = receptor.size(2)//3
            neg_dr.data = neg_dr.data.clamp_(-receptor.size(2)+clamp_offset, receptor.size(2)-clamp_offset)
            neg_alpha.data = neg_alpha.data.clamp_(-np.pi, np.pi)

            # neg_dr.data += noise_dr.normal_(0, self.sig_dr)
            # neg_alpha.data += noise_alpha.normal_(0, self.sig_alpha)
            # neg_dr.data = neg_dr.data.clamp_(-rec_feat.size(2), rec_feat.size(2))
            # neg_alpha.data = neg_alpha.data.clamp_(-np.pi, np.pi)
            # with torch.no_grad():
                # print(neg_alpha.clamp_(-np.pi, np.pi))
                # neg_dr_out = neg_dr.data.clone()
                # neg_alpha_out = neg_alpha.data.clone()
                # print('transform inside LD')
                # print(neg_alpha, neg_dr)
                # print(neg_alpha_out, neg_dr_out)

            # print('inside LD')
            # print(neg_alpha, neg_dr)
            # print(neg_alpha.data, neg_dr.data)

            if self.debug:
                if i == 0 or i == 1:
                    print('After RandomForce LD', i)
                    print(neg_alpha)
                    print(neg_dr)

        # if self.FI:
        #     return neg_alpha_out.detach(), neg_dr_out.detach(), lastN_alpha, lastN_dr, lastN_neg_out
        # else:
        #     return neg_alpha.detach(), neg_dr.detach()

        # self.requires_grad(True)
        self.EBMdocker.train()
        # return neg_alpha.detach(), neg_dr.detach()

        return neg_alpha.clone().detach(), neg_dr.clone().detach(), FFT_score

    def requires_grad(self, flag=True):
        parameters = self.EBMdocker.parameters()
        for p in parameters:
            p.requires_grad = flag

    @staticmethod
    def check_gradients(model, param=None):
        for n, p in model.named_parameters():
            if param and param in str(n):
                print('Name', n, '\nParam', p, '\nGradient', p.grad)
                return
            if not param:
                print('Name', n, '\nParam', p, '\nGradient', p.grad)



if __name__ == "__main__":
    pass
