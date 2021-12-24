import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from DeepProteinDocking2D.Models.EBM.TorchDockingFFT import TorchDockingFFT
from e2cnn import nn as enn
from e2cnn import gspaces
import random

from DeepProteinDocking2D.EQScoring import EQScoringModel, EQDockerGPU
from DeepProteinDocking2D.Convolution import ProteinConv2D

class EBMDocking(nn.Module):
    def __init__(self):
        super(EBMDocking, self).__init__()
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
        num_buf_samples = 880
        device = 'cuda'
        num_samples = 1
        weight = 1.0
        step_size = 10.0
        sample_steps = 100
        self.buffer = SampleBuffer(num_buf_samples)
        self.buffer2 = SampleBuffer(num_buf_samples)

        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.weight = weight
        self.step_size = step_size
        self.device = device

        self.scorer = nn.Sequential(
            nn.MaxPool2d(100),
            nn.Linear(1, 1, bias=False),
        )

        self.softmax = torch.nn.Softmax(dim=0)
        # self.docker = TorchDockingFFT()
        self.conv = ProteinConv2D()

    def requires_grad(self, flag=True):
        parameters = EBMDocking().parameters()
        for p in parameters:
            p.requires_grad = flag

    def docked_translations(self, rec_feat, lig_feat, rotation):
        scores = TorchDockingFFT(rotation=rotation).dock_global(
            rec_feat,
            lig_feat,
            weight_bound=self.boundW,
            weight_crossterm1=self.crosstermW1,
            weight_crossterm2=self.crosstermW2,
            weight_bulk=self.bulkW
        )

        pred_rot, pred_txy, best_score = TorchDockingFFT(rotation=rotation).extract_transform(scores)

        return pred_txy, best_score

    def forward(self, data, plotting=False):
        receptor, ligand, gt_txy, gt_rot, pos_idx = data

        receptor_geomT = enn.GeometricTensor(receptor.unsqueeze(0), self.feat_type_in1)
        ligand_geomT = enn.GeometricTensor(ligand.unsqueeze(0), self.feat_type_in1)

        rec_feat = self.netSE2(receptor_geomT).tensor.squeeze()
        lig_feat = self.netSE2(ligand_geomT).tensor.squeeze()

        L_p, L_n, L_n2, neg_alpha, neg_dr = self.step_parallel(pos_idx, receptor, ligand, rec_feat, lig_feat)

        return L_p, L_n, L_n2, neg_alpha, neg_dr

    def langevin(self, neg_alpha, neg_dr, rec_feat, lig_feat, sigma_dr=0.5, sigma_alpha=5.0):
        noise_alpha = torch.zeros_like(neg_alpha)
        noise_dr = torch.zeros_like(neg_dr)

        self.requires_grad(False)
        # EBMDocking().eval()

        neg_alpha.requires_grad_()
        neg_dr.requires_grad_()
        langevin_opt = torch.optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

        lastN_neg_out = []

        self.sample_steps = 1
        for i in range(self.sample_steps):
            langevin_opt.zero_grad()

            # neg_dr, neg_out = self.docked_translations(rec_feat, lig_feat, neg_alpha)
            # # print('feat stack shape',feat_stack.shape)
            # print(neg_dr, neg_out)
            # neg_out.mean().backward()
            #
            # print(neg_dr, neg_out)

            pos_repr, _, A = self.model.mult(rec_feat, lig_feat, neg_alpha, neg_dr)
            neg_out = self.model.scorer(pos_repr)
            neg_out.mean().backward()
            langevin_opt.step()

            # print(neg_alpha.shape, noise_alpha.normal_(0, sigma_alpha).shape)
            # print(neg_dr.shape, noise_dr.normal_(0, sigma_dr).shape)

            neg_dr.data += noise_dr.normal_(0, sigma_dr).squeeze().type(torch.long)
            neg_alpha.data += noise_alpha.normal_(0, sigma_alpha).squeeze().type(torch.long)

            neg_dr.data.clamp_(-rec_feat.size(2), rec_feat.size(2))
            neg_alpha.data.clamp_(-np.pi, np.pi)

            lastN_neg_out.append(neg_out.detach())

        return neg_alpha.detach(), neg_dr.detach()

    def step_parallel(self, pos_idx, pos_alpha, pos_dr, rec_feat, lig_feat, epoch=None, train=True):

        neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=self.num_samples)
        neg_alpha2, neg_dr2 = self.buffer2.get(pos_idx, num_samples=self.num_samples)

        neg_rec_feat = rec_feat
        neg_lig_feat = lig_feat
        # pos_rec_feat = rec_feat
        # pos_lig_feat = lig_feat

        ## EBM IP parallel model
        neg_alpha, neg_dr = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(), neg_lig_feat.detach(),
                                          sigma_dr=0.05, sigma_alpha=0.5)
        neg_alpha2, neg_dr2 = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(), neg_lig_feat.detach(),
                                             sigma_dr=0.5, sigma_alpha=5)

        # print('neg_alpha, neg_dr', neg_alpha, neg_dr)
        # print('neg_alpha, neg_dr grad', neg_alpha.grad, neg_dr.grad)

        self.requires_grad(True)
        # self.model.train()
        # self.model.zero_grad()

        # pos_dr, pos_out = self.docked_translations(rec_feat, lig_feat, neg_alpha)
        # L_p = (pos_out + self.weight * pos_out ** 2).mean()
        #
        # neg_dr, neg_out = self.docked_translations(rec_feat, lig_feat, neg_alpha)
        # L_n = (-neg_out + self.weight * pos_out ** 2).mean()
        #
        # neg_dr2, neg_out2 = self.docked_translations(rec_feat, lig_feat, neg_alpha2)
        # L_n2 = (-neg_out2 + self.weight * pos_out ** 2).mean()

        pos_out, _, _ = self.model.mult(rec_feat, lig_feat, pos_alpha, pos_dr)
        pos_out = self.model.scorer(pos_out)
        L_p = (pos_out + self.weight * pos_out ** 2).mean()
        # L_p = (pos_out).mean()
        neg_out, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr)
        neg_out = self.model.scorer(neg_out)
        L_n = (-neg_out + self.weight * neg_out ** 2).mean()
        # L_n = (-neg_out).mean()
        neg_out2, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha2, neg_dr2)
        neg_out2 = self.model.scorer(neg_out2)
        L_n2 = (-neg_out2 + self.weight * neg_out2 ** 2).mean()
        # L_n2 = (-neg_out2).mean()

        self.buffer.push(neg_alpha.unsqueeze(0), neg_dr.unsqueeze(0), pos_idx)
        self.buffer2.push(neg_alpha2.unsqueeze(0), neg_dr2.unsqueeze(0), pos_idx)

        return L_p, L_n, L_n2, neg_alpha, neg_dr

class SampleBuffer:
    def __init__(self, num_samples, max_pos=100):
        self.num_samples = num_samples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_samples):
            self.buffer[i] = []

    def __len__(self, i):
        return len(self.buffer[i])

    def push(self, alphas, drs, index):
        alphas = alphas.detach().to(device='cpu')
        drs = drs.detach().to(device='cpu')

        for alpha, dr, idx in zip(alphas, drs, index):
            i = idx.item()
            self.buffer[i].append((alpha, dr))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)

    def get(self, index, num_samples, device='cuda'):
        alphas = []
        drs = []
        for idx in index:
            i = idx.item()
            if len(self.buffer[i]) >= num_samples and random.randint(0, 10) < 7:
                # print('if statement')
                # print('len buffer', len(self.buffer[i]))
                lst = random.choices(self.buffer[i], k=num_samples)
                alpha = list(map(lambda x: x[0], lst))
                dr = list(map(lambda x: x[1], lst))
                alphas.append(torch.stack(alpha, dim=0))
                drs.append(torch.stack(dr, dim=0))
            else:
                # print('else statement')
                # print('len buffer', len(self.buffer[i]), self.buffer[i])
                alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
                dr = torch.rand(num_samples, 2) * 50.0 - 25.0
                alphas.append(alpha)
                drs.append(dr)
            # print('\nalpha', alpha)
            # print('dr', dr)

        alphas = torch.stack(alphas, dim=0).to(device=device)
        drs = torch.stack(drs, dim=0).to(device=device)

        return alphas, drs


if __name__ == '__main__':
    print('works')
    print(EBMDocking())
    print(list(EBMDocking().parameters()))
