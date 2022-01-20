import torch
from torch import optim
import torch.nn as nn
import numpy as np

from DeepProteinDocking2D.Models import EQDockerGPU, EQScoringModel, EQRepresentation
from DeepProteinDocking2D.Models.Convolution import ProteinConv2D
# from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
from plot_EBM import EBMPlotter

import random

from matplotlib import pylab as plt
import os
import sys
sys.path.append('/home/sb1638/')



class EBMInteractionModel(nn.Module):
    def __init__(self):
        super(EBMInteractionModel, self).__init__()

        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
        # self.F_1 = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, sampling, sampling2=None, L_n=None, L_n2=None, hotcold=False):

        if hotcold:
            # print(L_n, L_n2)
            # sampling.append(L_n)
            E1 = torch.stack(sampling, dim=0)
            F1 = -torch.logsumexp(-E1, dim=(0, 1, 2))

            sampling2.append(L_n2)
            E2 = torch.stack(sampling2, dim=0)
            F2 = -torch.logsumexp(-E2, dim=(0, 1, 2))

            # deltaF = (F1 - self.F_0) + (F2 - self.F_1)
            deltaF = (F1 + F2) - self.F_0
            pred_interact = torch.sigmoid(-deltaF)

            with torch.no_grad():
                print('Fs', F1.item(), F2.item())
                print('\n(deltaF - F_0): ', deltaF.item())
                print('F_0: ', self.F_0.item(), 'F_0 grad', self.F_0.grad)
                # print('F_1: ', self.F_1.item(), 'F_1 grad', self.F_1.grad)

        else:
            if L_n:
                sampling.append(L_n)
            E = torch.stack(sampling, dim=0)
            F = -torch.logsumexp(-E, dim=(0, 1, 2))

            deltaF = F - self.F_0
            pred_interact = torch.sigmoid(-deltaF)

            with torch.no_grad():
                print('F', F.item())
                print('\n(deltaF - F_0): ', deltaF.item())
                print('F_0: ', self.F_0.item(), 'F_0 grad', self.F_0.grad)

        return pred_interact.squeeze(), deltaF.squeeze()


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
                lst = random.choices(self.buffer[i], k=num_samples)
                alpha = list(map(lambda x: x[0], lst))
                dr = list(map(lambda x: x[1], lst))
                alphas.append(torch.stack(alpha, dim=0))
                drs.append(torch.stack(dr, dim=0))
            else:
                alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
                dr = torch.rand(num_samples, 2) * 50.0 - 25.0
                alphas.append(alpha)
                drs.append(dr)
        # print('\nalpha', alpha)
        # print('dr', dr)

        alphas = torch.stack(alphas, dim=0).to(device=device)
        drs = torch.stack(drs, dim=0).to(device=device)

        return alphas, drs


class EBMTrainer:

    def __init__(self, model, optimizer, num_buf_samples=10, device='cuda', num_samples=10, weight=1.0, step_size=10.0,
                 sample_steps=100,
                 global_step=True, add_positive=True, FI=False, experiment=None, path_pretrain=None):
        self.model = model
        self.optimizer = optimizer

        self.buffer = SampleBuffer(num_buf_samples)
        self.buffer2 = SampleBuffer(num_buf_samples)
        self.global_step = global_step
        self.add_positive = add_positive

        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.weight = weight
        self.step_size = step_size
        self.step_size = 100
        self.device = device

        self.plot_idx = 0
        self.conv = ProteinConv2D()

        self.sig_dr = 0.05
        self.sig_alpha = 0.5

        self.docker = EQDockerGPU(EQScoringModel(repr=None).to(device='cuda'))

        self.plot_freq = 20

        self.FI = FI
        if self.FI:
            print("LOAD FImodel ONCE??????")
            ##### load blank model and optimizer, once
            # lr_interaction = 10 ** -1
            # self.interaction_model = EBMInteractionModel().to(device=0)
            # self.optimizer_interaction = optim.Adam(self.interaction_model.parameters(), lr=lr_interaction, betas=(0.0, 0.999))
            # self.plot_freq = 100
            # self.sig_dr = 5
            # self.sig_alpha = 5
            # self.sig_hotweight = 5
            lr_interaction = 10 ** -2
            self.interaction_model = EBMBFInteractionModel().to(device=0)
            self.optimizer_interaction = optim.Adam(self.interaction_model.parameters(), lr=lr_interaction, betas=(0.0, 0.999))
            self.plot_freq = 100

        self.experiment = experiment
        self.path_IP = '../../EBM_figs/IP_figs/' + self.experiment
        self.path_FI = '../../EBM_figs/FI_figs/' + self.experiment
        try:
            if 'IP' in self.experiment:
                os.mkdir(self.path_IP)
            if 'FI' in self.experiment:
                os.mkdir(self.path_FI)
        except:
            print('dir already exists')

        self.debug = False
        self.train = False
        # self.pretrain_init = False

    def requires_grad(self, flag=True):
        parameters = self.model.parameters()
        for p in parameters:
            p.requires_grad = flag

    @staticmethod
    def check_gradients(model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                print('Name', n, '\nParam', p, '\nGradient', p.grad)

    def load_checkpoint(self, path):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = torch.load(path)
        raw_model.load_state_dict(checkpoint)

    def dock_spatial(self, rec_repr, lig_repr):
        translations = self.conv(rec_repr, lig_repr)

        batch_size = translations.size(0)
        num_features = translations.size(1)
        L = translations.size(2)

        translations = translations.view(batch_size, num_features, L * L)
        translations = translations.transpose(1, 2).contiguous().view(batch_size * L * L, num_features)
        scores = self.model.scorer(translations).squeeze()
        scores = scores.view(batch_size, L, L)

        minval_y, ind_y = torch.min(scores, dim=2, keepdim=False)
        minval_x, ind_x = torch.min(minval_y, dim=1)
        x = ind_x
        y = ind_y[torch.arange(batch_size), ind_x]

        x -= int(L / 2)
        y -= int(L / 2)

        return torch.stack([x, y], dim=1).to(dtype=lig_repr.dtype, device=lig_repr.device)

    def rotate(self, repr, angle):
        alpha = angle.detach()
        T0 = torch.cat([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
        T1 = torch.cat([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
        R = torch.stack([T0, T1], dim=1)
        curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
        return nn.functional.grid_sample(repr, curr_grid, align_corners=True)

    def langevin(self, neg_alpha, neg_dr, rec_feat, lig_feat, neg_idx, temperature='cold'):
        noise_alpha = torch.zeros_like(neg_alpha)
        noise_dr = torch.zeros_like(neg_dr)

        self.requires_grad(False)
        self.model.eval()

        # if self.FI:
        ## Sid global step
        with torch.no_grad():
            translations = self.docker.dock_global(rec_feat, lig_feat)
            scores = self.docker.score(translations)
            score, rotation, translation = self.docker.get_conformation(scores)
            neg_alpha = rotation.unsqueeze(0).unsqueeze(0).cuda()
            neg_dr = translation.unsqueeze(0).cuda()

        if self.global_step:
            with torch.no_grad():
                rlig_feat = self.rotate(lig_feat, neg_alpha)
                neg_dr = self.dock_spatial(rec_feat, rlig_feat)

        neg_alpha.requires_grad_()
        neg_dr.requires_grad_()
        langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

        if not self.train:
            self.sig_dr = 0.0
            self.sig_alpha = 0.0
            self.sig_hotweight = 0.0

        if temperature == 'hot':
            self.sig_dr = 0.5
            self.sig_alpha = 5

        lastN_neg_out = []
        lastN_alpha = []
        lastN_dr = []

        for i in range(self.sample_steps):
            langevin_opt.zero_grad()

            pos_repr, _, A = self.model.mult(rec_feat, lig_feat, neg_alpha, neg_dr)
            neg_out = self.model.scorer(pos_repr)
            neg_out.mean().backward()

            langevin_opt.step()

            # neg_dr.data += noise_dr.normal_(0, self.sig_dr)
            # neg_alpha.data += noise_alpha.normal_(0, self.sig_alpha)

            neg_dr = neg_dr + noise_dr.normal_(0, self.sig_dr)
            neg_alpha = neg_alpha + noise_alpha.normal_(0, self.sig_alpha)

            neg_dr.data.clamp_(-rec_feat.size(2), rec_feat.size(2))
            neg_alpha.data.clamp_(-np.pi, np.pi)

            # print(neg_alpha)
            # print(neg_dr)
            # print(neg_out)
            lastN_neg_out.append(neg_out.detach())
            lastN_alpha.append(neg_alpha)
            lastN_dr.append(neg_dr)

        if self.FI:
            return lastN_alpha, lastN_dr, lastN_neg_out, rotation.unsqueeze(0).unsqueeze(0).cuda(), translation.unsqueeze(0).cuda(), scores
            # return neg_alpha.detach(), neg_dr.detach(), lastN_neg_out, rotation, translation
        else:
            return neg_alpha.detach(), neg_dr.detach()

    def step_parallel(self, data, epoch=None, train=True):
        gt_interact = None
        pos_alpha = None
        pos_dr = None
        self.train = train
        if self.FI:
            receptor, ligand, gt_interact, pos_idx = data
            pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
            pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
            gt_interact = gt_interact.to(device=self.device, dtype=torch.float32)
            pos_idx = pos_idx.to(device=self.device, dtype=torch.long)
        else:
            receptor, ligand, translation, rotation, pos_idx = data
            pos_rec = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
            pos_lig = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
            pos_alpha = rotation.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
            pos_dr = translation.to(device=self.device, dtype=torch.float32)

        batch_size = pos_rec.size(0)
        num_features = pos_rec.size(1)
        L = pos_rec.size(2)

        neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=self.num_samples)
        neg_alpha2, neg_dr2 = self.buffer2.get(pos_idx, num_samples=self.num_samples)

        neg_rec = pos_rec.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size * self.num_samples,
                                                                                     num_features, L, L)
        neg_lig = pos_lig.unsqueeze(dim=1).repeat(1, self.num_samples, 1, 1, 1).view(batch_size * self.num_samples,
                                                                                     num_features, L, L)
        neg_idx = pos_idx.unsqueeze(dim=1).repeat(1, self.num_samples).view(batch_size * self.num_samples)
        neg_alpha = neg_alpha.view(batch_size * self.num_samples, -1)
        neg_dr = neg_dr.view(batch_size * self.num_samples, -1)
        neg_alpha2 = neg_alpha2.view(batch_size * self.num_samples, -1)
        neg_dr2 = neg_dr2.view(batch_size * self.num_samples, -1)

        neg_rec_feat = self.model.repr(neg_rec).tensor
        neg_lig_feat = self.model.repr(neg_lig).tensor
        pos_rec_feat = self.model.repr(pos_rec).tensor
        pos_lig_feat = self.model.repr(pos_lig).tensor

        if self.FI:
            return self.FI_prediction(neg_rec_feat, neg_lig_feat, pos_rec_feat, pos_lig_feat, neg_alpha, neg_dr, neg_alpha2, neg_dr2,
                                               pos_idx, neg_idx, receptor, ligand, gt_interact, epoch)
        else:

            return self.IP_prediction(pos_rec_feat, pos_lig_feat, neg_rec_feat, neg_lig_feat,
                                        pos_alpha, pos_dr, neg_alpha, neg_dr, neg_alpha2, neg_dr2,
                                        pos_idx, neg_idx, receptor, ligand, epoch)

    def IP_prediction(self, pos_rec_feat, pos_lig_feat, neg_rec_feat, neg_lig_feat, pos_alpha, pos_dr, neg_alpha,
                        neg_dr, neg_alpha2, neg_dr2, pos_idx, neg_idx, receptor, ligand, epoch):

        neg_alpha, neg_dr = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx,
                                          'cold')
        neg_alpha2, neg_dr2 = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx,
                                            'hot')

        if self.train:
            self.model.train()
            self.requires_grad(True)
            self.model.zero_grad()
        else:
            self.model.eval()

        pos_out, _, _ = self.model.mult(pos_rec_feat, pos_lig_feat, pos_alpha, pos_dr)
        pos_out = self.model.scorer(pos_out)
        L_p = (pos_out + self.weight * pos_out ** 2).mean()
        # L_p = pos_out.mean()
        neg_out, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha, neg_dr)
        neg_out = self.model.scorer(neg_out)
        L_n = (-neg_out + self.weight * neg_out ** 2).mean()
        # L_n = -neg_out.mean()
        neg_out2, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha2, neg_dr2)
        neg_out2 = self.model.scorer(neg_out2)
        L_n2 = (-neg_out2 + self.weight * neg_out2 ** 2).mean()
        # L_n2 = -neg_out2.mean()

        L_n = (L_n + L_n2).mean()
        loss = L_p + L_n

        # loss = L_p + L_n
        # l1_loss = torch.nn.L1Loss()
        # lossR = l1_loss(pos_alpha, neg_alpha)
        # L_r = (lossR + self.weight * lossR ** 2).mean()
        # lossT = l1_loss(pos_dr, neg_dr)
        # L_t = (lossT + self.weight * lossT ** 2).mean()
        # lossR2 = l1_loss(pos_alpha, neg_alpha2)
        # L_r2 = (lossR2 + self.weight * lossR2 ** 2).mean()
        # lossT2 = l1_loss(pos_dr, neg_dr2)
        # L_t2 = (lossT2 + self.weight * lossT2 ** 2).mean()
        # loss = loss + L_r + L_t + L_r2 + L_t2

        if self.train:
            loss.backward()
            self.optimizer.step()

        if not self.train and self.debug:
            with torch.no_grad():
                # print('\nLearned hot sim contribution', self.hotweight.item())
                # print('\nL_p, L_n, \n', L_p.item(), L_n.item())
                # print('Loss\n', loss.item())

                filename = self.path_IP + '/IPenergyandpose_epoch' + str(epoch) + '_example' + str(pos_idx.item())
                EBMPlotter(self.model).plot_energy_and_pose(pos_idx, L_p, L_n, epoch, receptor, ligand, pos_alpha, pos_dr,
                                          neg_alpha, neg_dr, filename)
                filename = self.path_IP + '/IPfeats_epoch' + str(epoch) + '_example' + str(pos_idx.item())
                EBMPlotter(self.model).plot_feats(neg_rec_feat, neg_lig_feat, epoch, pos_idx, filename)

        # never add postive for step parallel and 1D LD buffer
        if self.add_positive:
            # print('AP')
            self.buffer.push(pos_alpha, pos_dr, pos_idx)
            self.buffer2.push(pos_alpha, pos_dr, pos_idx)

        self.buffer.push(neg_alpha, neg_dr, neg_idx)
        self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)

        if self.debug:
            print('checking gradients')
            self.check_gradients(self.model)

        if self.train:
            return {"Loss": loss.item()}
        else:
            return {"Loss": loss.item()}, neg_alpha, neg_dr

    def FI_prediction(self, neg_rec_feat, neg_lig_feat, pos_rec_feat, pos_lig_feat, neg_alpha, neg_dr, neg_alpha2, neg_dr2, pos_idx,
                               neg_idx, receptor, ligand, gt_interact, epoch):


        # #### working model with gradient back to CNN and scoring layer
        # with torch.no_grad():
        #     translations = self.docker.dock_global(pos_rec_feat, pos_lig_feat)
        #     scores = self.docker.score(translations)
        #     Energies = -scores
        #     score, rotation, translation = self.docker.get_conformation(Energies)
        #     pos_alpha = rotation.unsqueeze(0).unsqueeze(0).cuda()
        #     pos_dr = translation.unsqueeze(0).cuda()
        #
        #     # print(Energies.shape)
        #     print('\nEnergy max and min\n')
        #     print(torch.max(Energies), torch.min(Energies))
        #
        # E_best_pos, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, pos_alpha, pos_dr)
        # E_best_pos = self.model.scorer(E_best_pos)
        # E_best_pos = E_best_pos
        #
        # pred_interact, deltaF = self.interaction_model(Energies, E_best_pos)
        # #### working model ^^^^

        neg_alpha_list, neg_dr_list, lastN_E_cold, rotation, translation, scores = self.langevin(neg_alpha, neg_dr,
                                                        neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx, 'cold')
        # , rotation, translation, scores
        if self.train:
            self.model.train()
            self.interaction_model.train()
            self.requires_grad(True)
            self.model.zero_grad()
            # self.interaction_model.zero_grad()
        else:
            self.model.eval()

        lastN_E_grad = []
        for i in range(len(lastN_E_cold)):
            # print(neg_alpha_list[i])
            # print(neg_dr_list[i])
            # print(lastN_E_cold[i])
            E_pred_neg, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha_list[i], neg_dr_list[i])
            E_pred_neg = self.model.scorer(E_pred_neg)
            lastN_E_grad.append(E_pred_neg)

        E_best_pos, _, _ = self.model.mult(pos_rec_feat, pos_lig_feat, rotation, translation)
        E_best_pos = self.model.scorer(E_best_pos)
        E_best_pos = E_best_pos

        Energies = torch.stack(lastN_E_grad, dim=0)
        with torch.no_grad():
            print(Energies.shape)
            print('\nEnergy max and min\n')
            print(torch.max(Energies), torch.min(Energies))
            print('E_best', E_best_pos.item())
        pred_interact, deltaF = self.interaction_model(Energies-E_best_pos)

        # pred_interact, deltaF = self.interaction_model(lastN_E_grad)


        # pos_alpha = rotation.unsqueeze(0).unsqueeze(0).cuda()
        # pos_dr = translation.unsqueeze(0).cuda()
        # Energies = torch.stack(lastN_E_grad, dim=0)

        # E_best_pos, _, _ = self.model.mult(pos_rec_feat, pos_lig_feat, pos_alpha, pos_dr)
        # E_best_pos = self.model.scorer(E_best_pos)
        # E_best_pos = E_best_pos

        # pred_interact, deltaF = self.interaction_model(-Energies, E_best_pos)

        if self.train:
            with torch.no_grad():
                filename_feats = self.path_FI + '/Feats_epoch' + str(epoch) + '_' + str(
                    self.sample_steps) + '_' + str(pos_idx.item())
                EBMPlotter(self.model).plot_feats(neg_rec_feat, neg_lig_feat, epoch, pos_idx, filename_feats)
                filename_pose = self.path_FI + '/Pose_epoch' + str(epoch) + '_' + str(
                    self.sample_steps) + '_' + str(pos_idx.item())
                EBMPlotter(self.model).plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD', filename_pose,
                               pos_idx, epoch,
                               # gt_rot=rotation.detach().cpu().numpy(),
                               # gt_txy=translation.detach().cpu().numpy(),
                               pred_interact=pred_interact.item(),
                               gt_interact=gt_interact.item())

        if self.train:
            BCEloss = torch.nn.BCELoss()
            l1_loss = torch.nn.L1Loss()

            w = 10 ** -5
            L_reg = w * l1_loss(deltaF.squeeze(), torch.zeros(1).squeeze().cuda())
            loss = BCEloss(pred_interact.squeeze(), gt_interact.squeeze().cuda()) + L_reg #+ EBMloss #+ l1_loss(E_best_pos, E_pred_neg)

            loss.backward()
            print('\n PREDICTED', pred_interact.item(), '; GROUND TRUTH', gt_interact.item())
            self.optimizer.step()
            self.optimizer_interaction.step()
            # self.optimizer_interaction.zero_grad()
            self.buffer.push(neg_alpha, neg_dr, neg_idx)
            self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)

            if self.debug:
                print('checking gradients')
                print('pretrain model')
                self.check_gradients(self.model)
                print('interaction model')
                self.check_gradients(self.interaction_model)

            return {"Loss": loss.item()}

        else:
            threshold = 0.5
            TP, FP, TN, FN = 0, 0, 0, 0
            p = pred_interact.item()
            a = gt_interact.item()
            if p >= threshold and a >= threshold:
                TP += 1
            elif p >= threshold > a:
                FP += 1
            elif p < threshold <= a:
                FN += 1
            elif p < threshold and a < threshold:
                TN += 1
            # print('returning', TP, FP, TN, FN)
            print('\n PREDICTED', pred_interact.item(), '; GROUND TRUTH', gt_interact.item())
            return TP, FP, TN, FN, pred_interact.squeeze() - gt_interact.squeeze().cuda()


# class EBMBFInteractionModel(nn.Module):
#     def __init__(self):
#         super(EBMBFInteractionModel, self).__init__()
#
#         self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))
#
#     def forward(self, sampling):
#         E = torch.stack(sampling, dim=0)
#         deltaF = -torch.logsumexp(-E, dim=(0, 1, 2)) - self.F_0
#         pred_interact = torch.sigmoid(-deltaF)
#         # deltaF = E.mean() - self.F_0
#         # pred_interact = torch.sigmoid(-deltaF)
#
#         with torch.no_grad():
#             # print('E_best', E_best.item())
#             print('\n(deltaF - F_0): ', deltaF.item())
#             print('F_0: ', self.F_0.item(), 'F_0 grad', self.F_0.grad)
#
#         return pred_interact.squeeze(), deltaF.squeeze()

class EBMBFInteractionModel(nn.Module):
    def __init__(self):
        super(EBMBFInteractionModel, self).__init__()

        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, E):
        deltaF = -torch.logsumexp(-E, dim=(0, 1, 2)) - self.F_0
        pred_interact = torch.sigmoid(-deltaF)

        with torch.no_grad():
            print('\n(deltaF - F_0): ', deltaF.item())
            print('F_0: ', self.F_0.item(), 'F_0 grad', self.F_0.grad)

        return pred_interact.squeeze(), deltaF.squeeze()


# neg_alpha, neg_dr, lastN_E_cold = self.langevin(neg_alpha, neg_dr, neg_rec_feat.detach(),
        #                                                      neg_lig_feat.detach(), neg_idx, 'cold')
        #
        # E_pred_neg, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha, pos_dr)
        # E_pred_neg = self.model.scorer(E_pred_neg)
        # E_pred_neg = E_pred_neg
        #
        # neg_alpha2, neg_dr2, lastN_E_cold = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(),
        #                                                      neg_lig_feat.detach(), neg_idx, 'hot')
        #
        # E_pred_neg2, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha2, pos_dr)
        # E_pred_neg2 = self.model.scorer(E_pred_neg2)
        # E_pred_neg2 =E_pred_neg2
        #
        # # EBMloss = E_best_pos + (E_pred_neg + E_pred_neg2).mean()
        # print('E_pred', E_pred_neg.item())
        # print('E_pred2', E_pred_neg2.item())

        # neg_alpha2, neg_dr2, lastN_E_hot = self.langevin(neg_alpha2, neg_dr2, neg_rec_feat.detach(),
        #                                                      neg_lig_feat.detach(), neg_idx, 'hot')
        #
        # E_pred_neg2, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha2, neg_dr2)
        # E_pred_neg2 = E_pred_neg2
        # E_pred_neg2 = self.model.scorer(E_pred_neg2)
        #
        # print('E_pred', E_pred_neg.item())
        # print('E_pred2', E_pred_neg2.item())
