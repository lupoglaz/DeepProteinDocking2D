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
# from torch.autograd import Function
# import torch.nn.functional as F
# from e2cnn import nn as enn
# from e2cnn import gspaces

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
    def __init__(self, num_angles=1):
        super(DockerEBM, self).__init__()
        self.num_angles = num_angles
        self.docker = BruteForceDocking(dim=100, num_angles=self.num_angles)
        self.dockingFFT = TorchDockingFFT(num_angles=self.num_angles, angle=None, swap_plot_quadrants=False)

    def forward(self, receptor, ligand, rotation, plot_count=1, stream_name='trainset', plotting=False):
        if 'trainset' not in stream_name:
            training = False
        else: training = True
        FFT_score = self.docker.forward(receptor, ligand, angle=rotation, plotting=plotting, training=training, plot_count=plot_count, stream_name=stream_name)
        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
            deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)
            if self.num_angles == 1:
                best_score = FFT_score[pred_txy[0], pred_txy[1]]
            else:
                best_score = FFT_score[deg_index_rot, pred_txy[0], pred_txy[1]]
        minE = -best_score
        return minE, pred_txy, pred_rot, FFT_score



class EnergyBasedModel(nn.Module):
    def __init__(self, num_angles=1, device='cuda', num_samples=1, weight=1.0, step_size=1, sample_steps=1, experiment=None):
        super(EnergyBasedModel, self).__init__()
        self.debug = False
        self.num_angles = num_angles

        self.EBMdocker = DockerEBM(num_angles=self.num_angles)

        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.weight = weight
        self.step_size = step_size
        self.device = device

        self.plot_idx = 0

        self.experiment = experiment

    def forward(self, neg_alpha, neg_dr, receptor, ligand, temperature='cold', plot_count=1, stream_name='trainset'):

        noise_alpha = torch.zeros_like(neg_alpha)
        # noise_dr = torch.zeros_like(neg_dr)

        self.EBMdocker.eval()

        neg_alpha.requires_grad_()
        # neg_dr.requires_grad_()
        # langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)
        langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)

        if temperature == 'cold':
            # self.sig_dr = 0.05
            # self.sig_alpha = 0.5
            self.sig_alpha = 6.28

        if temperature == 'hot':
            # self.sig_dr = 0.5
            self.sig_alpha = 5

        # if self.sample_steps == 0:
        #     print('evaluating with brute force')
        #     minE, neg_dr, neg_alpha, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=True)
        #     return neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score, minE
        #
        # if 'trainset' not in stream_name:
        #     # print('eval')
        #     self.sample_steps = 1

        for i in range(self.sample_steps):
            plotting = False
            if i == self.sample_steps - 1:
                plotting = True

            langevin_opt.zero_grad()

            # neg_alpha = neg_alpha + noise_alpha.normal_(0, self.sig_alpha)
            # # neg_alpha.data = neg_alpha.data.clamp_(-np.pi, np.pi)
            # # neg_dr = neg_dr + noise_dr.normal_(0, self.sig_dr)
            # # clamp_offset = receptor.size(2)//3
            # # neg_dr.data = neg_dr.data.clamp_(-receptor.size(2)+clamp_offset, receptor.size(2)-clamp_offset)

            minE, neg_dr, pred_rot, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)

            # print('negdr in LD', temperature, neg_dr)
            # print('negalpha in LD', temperature, neg_alpha)
            minE.mean().backward(retain_graph=True)

            # minE.mean().backward()
            langevin_opt.step()

            neg_alpha = neg_alpha + noise_alpha.normal_(0, self.sig_alpha)
            # neg_alpha.data = neg_alpha.data.clamp_(-np.pi, np.pi)
            # neg_dr = neg_dr + noise_dr.normal_(0, self.sig_dr)
            # clamp_offset = receptor.size(2)//3
            # neg_dr.data = neg_dr.data.clamp_(-receptor.size(2)+clamp_offset, receptor.size(2)-clamp_offset)

        self.EBMdocker.train()

        return neg_alpha.clone(), neg_dr.clone(), FFT_score, minE

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
