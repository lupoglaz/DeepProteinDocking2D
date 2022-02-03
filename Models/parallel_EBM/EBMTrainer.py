import torch
from torch import optim
import torch.nn as nn
import numpy as np

from DeepProteinDocking2D.Models import EQDockerGPU, EQScoringModel, EQRepresentation
from DeepProteinDocking2D.Models.Convolution import ProteinConv2D
from plot_EBM import EBMPlotter

import random

from matplotlib import pylab as plt
import os
import sys
sys.path.append('/home/sb1638/')


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

    def get(self, index, num_samples, device='cuda', last_transform=None):
        alphas = []
        drs = []
        for idx in index:
            i = idx.item()
            if len(self.buffer[i]) >= num_samples:
                # print('buffer if')
                lst = random.choices(self.buffer[i], k=num_samples)
                alpha = list(map(lambda x: x[0], lst))
                dr = list(map(lambda x: x[1], lst))
                alphas.append(torch.stack(alpha, dim=0))
                drs.append(torch.stack(dr, dim=0))
            else:
                # print('buffer else')
                if last_transform:
                    alphas.append(last_transform[0])
                    drs.append(last_transform[1])
                else:
                    # alphas.append(torch.zeros(num_samples, 1))
                    # drs.append(torch.zeros(num_samples, 2))
                    alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
                    dr = torch.rand(num_samples, 2) * 50.0 - 25.0
                    alphas.append(alpha)
                    drs.append(dr)

            # if len(self.buffer[i]) >= num_samples and random.randint(0, 10) < 7:
            #     lst = random.choices(self.buffer[i], k=num_samples)
            #     alpha = list(map(lambda x: x[0], lst))
            #     dr = list(map(lambda x: x[1], lst))
            #     alphas.append(torch.stack(alpha, dim=0))
            #     drs.append(torch.stack(dr, dim=0))
            #     # print(alpha, dr)
            # else:
            #     alpha = torch.rand(num_samples, 1) * 2 * np.pi - np.pi
            #     dr = torch.rand(num_samples, 2) * 50.0 - 25.0
            #     alphas.append(alpha)
            #     drs.append(dr)
            # print('\ni', i)
            # print('buffer')
            # print(self.buffer[i])

        # print('\nalpha', alpha)
        # print('dr', dr)

        alphas = torch.stack(alphas, dim=0).to(device=device)
        drs = torch.stack(drs, dim=0).to(device=device)

        return alphas, drs


class EBMTrainer:

    def __init__(self, model, optimizer, num_buf_samples=10, device='cuda', num_samples=10, weight=1.0, step_size=10.0,
                 sample_steps=100,
                 global_step=True, add_positive=True, FI=False, experiment=None, path_pretrain=None):

        self.debug = False
        self.train = False
        self.BF_init = False
        self.Force_reg = True
        if self.Force_reg:
            self.k = torch.ones(1).cuda() * 1e-3

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
        self.device = device

        self.plot_idx = 0
        self.conv = ProteinConv2D()

        self.sig_dr = 0.05
        self.sig_alpha = 0.5

        # self.sig_dr = 0.5
        # self.sig_alpha = 5

        self.docker = EQDockerGPU(EQScoringModel(repr=None).to(device='cuda'))

        self.FI = FI
        if self.FI:
            # print("LOAD FImodel ONCE??????")
            ##### load blank model and optimizer, once
            lr_interaction = 10 ** -3
            self.interaction_model = EBMBFInteractionModel().to(device=0)
            self.optimizer_interaction = optim.Adam(self.interaction_model.parameters(), lr=lr_interaction, betas=(0.0, 0.999))

        self.experiment = experiment
        self.path_IP = '../../EBM_figs/IP_figs/' + self.experiment
        self.path_FI = '../../EBM_figs/FI_figs/' + self.experiment
        self.path_LD = '../../EBM_figs/FI_figs/' + self.experiment + '/LD_steps'

        try:
            if 'IP' in self.experiment:
                os.mkdir(self.path_IP)
            if 'FI' in self.experiment:
                os.mkdir(self.path_FI)
                os.mkdir(self.path_LD)
        except:
            print('dir already exists')


    def requires_grad(self, flag=True):
        parameters = self.model.parameters()
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

        if self.BF_init:
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
        self.k.requires_grad_()
        langevin_opt = optim.SGD([neg_alpha, neg_dr], lr=self.step_size, momentum=0.0)

        # if not self.train:
        #     self.sig_dr = 0.0
        #     self.sig_alpha = 0.0

        if temperature == 'hot':
            self.sig_dr = 0.5
            self.sig_alpha = 5
            # self.sig_dr = 10
            # self.sig_alpha = 20

        lastN_neg_out = []
        lastN_alpha = []
        lastN_dr = []

        for i in range(self.sample_steps):
            langevin_opt.zero_grad()

            pos_repr, _, A = self.model.mult(rec_feat, lig_feat, neg_alpha, neg_dr)
            neg_out = self.model.scorer(pos_repr)
            if self.Force_reg:
                # print(neg_alpha, neg_dr)
                # print(self.k)
                neg_out = neg_out + self.k * torch.sqrt(neg_dr.squeeze()[0]**2 + neg_dr.squeeze()[1]**2 + 1e-7)
            neg_out.mean().backward()

            langevin_opt.step()

            dr_noise = noise_dr.normal_(0, self.sig_dr)
            alpha_noise = noise_alpha.normal_(0, self.sig_alpha)
            neg_dr.data += dr_noise
            neg_alpha.data += alpha_noise

            # neg_dr.data = neg_dr.data.clamp_(-rec_feat.size(2), rec_feat.size(2))
            # neg_alpha.data = neg_alpha.data.clamp_(-np.pi, np.pi)

            with torch.no_grad():
                # print(neg_alpha.clamp_(-np.pi, np.pi))
                neg_dr_out = neg_dr.clone()#.clamp_(-rec_feat.size(2), rec_feat.size(2))
                neg_alpha_out = neg_alpha.clone()#.clamp_(-np.pi, np.pi)

            # print('inside LD')
            # print(neg_alpha, neg_dr)
            # print(neg_alpha.data, neg_dr.data)

            lastN_neg_out.append(neg_out.detach())
            lastN_alpha.append(neg_alpha_out.detach())
            lastN_dr.append(neg_dr_out.detach())

        if self.FI:
            return lastN_alpha, lastN_dr, lastN_neg_out
        else:
            return neg_alpha.detach(), neg_dr.detach()

    def step_parallel(self, data, epoch=None, train=True, last_transform=None):
        self.requires_grad(True)
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

        neg_alpha, neg_dr = self.buffer.get(pos_idx, num_samples=self.num_samples, last_transform=last_transform)
        neg_alpha2, neg_dr2 = self.buffer2.get(pos_idx, num_samples=self.num_samples, last_transform=last_transform)

        # print(neg_alpha, neg_dr)

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

        if self.train:
            loss.backward()
            self.optimizer.step()

        if not self.train and self.debug:
            with torch.no_grad():
                # print('\nLearned hot sim contribution', self.hotweight.item())
                # print('\nL_p, L_n, \n', L_p.item(), L_n.item())
                # print('Loss\n', loss.item())

                filename = self.path_IP + '/IPenergyandpose_epoch' + str(epoch+1) + '_example' + str(pos_idx.item())
                EBMPlotter(self.model).plot_energy_and_pose(pos_idx, L_p, L_n, epoch, receptor, ligand, pos_alpha, pos_dr,
                                          neg_alpha, neg_dr, filename)
                filename = self.path_IP + '/IPfeats_epoch' + str(epoch+1) + '_example' + str(pos_idx.item())
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

        # #### grad recompute model
        if self.debug:
            print("BEFORE LD")
            print(neg_alpha, neg_dr)
        neg_alpha_list_cold, neg_dr_list_cold, lastN_E_cold = self.langevin(neg_alpha, neg_dr,
                                                        neg_rec_feat.detach(), neg_rec_feat.detach(), neg_idx, 'cold')

        # neg_alpha_list_hot, neg_dr_list_hot, lastN_E_hot = self.langevin(neg_alpha2, neg_dr2,
        #                                                 neg_rec_feat.detach(), neg_lig_feat.detach(), neg_idx, 'hot')

        if self.debug:
            print("After LD")
            print(neg_alpha, neg_dr)

        if self.train:
            self.model.train()
            self.interaction_model.train()
            self.requires_grad(True)
            self.model.zero_grad()
            self.interaction_model.zero_grad()
        else:
            self.model.eval()

        ## no gradient
        # Energies = torch.stack(lastN_E_cold, dim=0)
        # pred_interact, deltaF = self.interaction_model(Energies)
        ## no gradient

        lastN_E_cold_grad = []
        for i in range(len(lastN_E_cold)):
            E_pred_neg_cold, _, _ = self.model.mult(neg_rec_feat, neg_lig_feat, neg_alpha_list_cold[i], neg_dr_list_cold[i])
            E_pred_neg_cold = self.model.scorer(E_pred_neg_cold)
            lastN_E_cold_grad.append(E_pred_neg_cold)
            # print(neg_dr_list_cold[i])
            # print(neg_alpha_list_cold[i])
            # print(E_pred_neg_cold)

        # lastN_E_hot_grad = []
        # for i in range(len(lastN_E_hot)):
        #     E_pred_neg_hot, _, _ = self.model.mult(neg_rec_feat, neg_rec_feat, neg_alpha_list_hot[i], neg_dr_list_hot[i])
        #     E_pred_neg_hot = self.model.scorer(E_pred_neg_hot)
        #     lastN_E_hot_grad.append(E_pred_neg_hot)

        Energies_cold_grad = torch.stack(lastN_E_cold_grad, dim=0)
        # Energies_hot_grad = torch.stack(lastN_E_hot_grad, dim=0)
        # pred_interact, deltaF = self.interaction_model(Energies_cold_grad, Energies_hot_grad)
        pred_interact, deltaF = self.interaction_model(Energies_cold_grad, Ehot=None)

        # Energies_cold = torch.stack(lastN_E_cold, dim=0)
        # Energies_hot = torch.stack(lastN_E_hot, dim=0)
        # pred_interact, deltaF = self.interaction_model(Energies_cold-Energies_hot)
        # #### grad recompute model

        if self.debug:
            with torch.no_grad():
                print('\nEnergy max and min')
                print(torch.max(Energies_cold_grad).item(), torch.min(Energies_cold_grad).item())
                # print(torch.max(Energies_hot_grad).item(), torch.min(Energies_hot_grad).item())
                # print(torch.max(Energies_cold).item(), torch.min(Energies_cold).item())
                # print(torch.max(Energies_hot).item(), torch.min(Energies_hot).item())

        if self.train:
            with torch.no_grad():
                filename_feats = self.path_FI + '/Feats_epoch' + str(epoch+1) + '_LDsteps' + str(
                    self.sample_steps) + '_sample' + str(pos_idx.item())
                EBMPlotter(self.model).plot_feats(neg_rec_feat, neg_lig_feat, epoch, pos_idx, filename_feats)
                filename_pose = self.path_FI + '/Pose_epoch' + str(epoch+1) + '_LDsteps' + str(
                    self.sample_steps) + '_sample' + str(pos_idx.item())
                EBMPlotter(self.model).plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD', filename_pose,
                               pos_idx, epoch,
                               # gt_rot=rotation.detach().cpu().numpy(),
                               # gt_txy=translation.detach().cpu().numpy(),
                               pred_interact=pred_interact.item(),
                               gt_interact=gt_interact.item())

                for i in range(len(lastN_E_cold)):
                    filename_pose = self.path_LD + '/sample' + str(pos_idx.item()) +'_epoch' + str(epoch+1) + '_LDstep'+str(i+1)
                    EBMPlotter(self.model).plot_pose(receptor, ligand, neg_alpha_list_cold[i].squeeze(),
                                                     neg_dr_list_cold[i].squeeze(), 'Pose after LD',
                                                     filename_pose,
                                                     pos_idx, epoch,
                                                     pred_interact=pred_interact.item(),
                                                     gt_interact=gt_interact.item(),
                                                     plot_LD=True)

        if self.train:
            BCEloss = torch.nn.BCELoss()
            loss = BCEloss(pred_interact.squeeze(), gt_interact.squeeze().cuda())

            # l1_loss = torch.nn.L1Loss()
            # w = 10 ** -5
            # L_reg = w * l1_loss(deltaF.squeeze(), torch.zeros(1).squeeze().cuda())
            # loss += L_reg

            # loss += loss_LpLn

            loss.backward()
            if self.debug:
                with torch.no_grad():
                    print('\n PREDICTED', pred_interact.item(), '; GROUND TRUTH', gt_interact.item())
                    if torch.round(pred_interact).item() == torch.round(gt_interact).item():
                        print(' GOOD')
                    else:
                        print(' BAD')

            self.optimizer.step()
            self.optimizer_interaction.step()
            self.buffer.push(neg_alpha, neg_dr, neg_idx)
            self.buffer2.push(neg_alpha2, neg_dr2, neg_idx)

            if self.debug:
                print('checking gradients')
                print('pretrain model')
                self.check_gradients(self.model, param='scorer')
                # self.check_gradients(self.model, param=None)
                print('interaction model')
                self.check_gradients(self.interaction_model, param=None)

            last_transform = (neg_alpha_list_cold[-1].squeeze(), neg_dr_list_cold[-1].squeeze())
            if self.debug:
                print(last_transform)

            return {"Loss": loss.item()}, last_transform

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
            if self.debug:
                with torch.no_grad():
                    print('\n PREDICTED', pred_interact.item(), '; GROUND TRUTH', gt_interact.item())
                    if torch.round(pred_interact).item() == torch.round(gt_interact).item():
                        print(' GOOD')
                    else:
                        print(' BAD')
            return TP, FP, TN, FN, pred_interact.squeeze() - gt_interact.squeeze().cuda()


class EBMBFInteractionModel(nn.Module):
    def __init__(self):
        super(EBMBFInteractionModel, self).__init__()

        self.F_0 = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, Ecold, Ehot=None):
        if Ehot:
            Fcold = -torch.logsumexp(-Ecold, dim=(0, 1, 2))
            Fhot = -torch.logsumexp(-Ehot, dim=(0, 1, 2))
            deltaF = Fcold + Fhot - self.F_0
        else:
            Fcold = -torch.logsumexp(-Ecold, dim=(0, 1, 2))
            deltaF = Fcold - self.F_0

        pred_interact = torch.sigmoid(-deltaF)

        # with torch.no_grad():
        #     print('\n(deltaF - F_0): ', deltaF.item())
        #     # print('F_0: ', self.F_0.item(), 'F_0 grad', self.F_0.grad)

        return pred_interact.squeeze(), deltaF.squeeze()
