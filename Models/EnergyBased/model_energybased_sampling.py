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


class Docker(nn.Module):
    def __init__(self, dockingFFT, num_angles=1, debug=False):
        super(Docker, self).__init__()
        self.num_angles = num_angles
        self.dim = 100
        self.dockingConv = BruteForceDocking(dim=self.dim, num_angles=self.num_angles, debug=debug)
        self.dockingFFT = dockingFFT

    def forward(self, receptor, ligand, rotation, plot_count=1, stream_name='trainset', plotting=False):
        if 'trainset' not in stream_name:
            training = False
        else: training = True

        if self.num_angles == 360:
            stream_name = 'BFeval_'+stream_name
        else:
            plotting=False

        FFT_score = self.dockingConv.forward(receptor, ligand, angle=rotation, plotting=plotting, training=training, plot_count=plot_count, stream_name=stream_name)

        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)

            if len(FFT_score.shape) > 2:
                deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)
                best_score = FFT_score[deg_index_rot, pred_txy[0], pred_txy[1]]
                if plotting and self.num_angles == 360 and plot_count % 10 == 0:
                    self.plot_rotE_surface(FFT_score, pred_txy, stream_name, plot_count)
            else:
                best_score = FFT_score[pred_txy[0], pred_txy[1]]

        lowest_energy = -best_score

        return lowest_energy, pred_rot, pred_txy, FFT_score

    def plot_rotE_surface(self, FFT_score, pred_txy, stream_name, plot_count):
        plt.close()
        mintxy_energies = []
        if self.num_angles == 1:
            minimumEnergy = -FFT_score[pred_txy[0], pred_txy[1]].detach().cpu()
            mintxy_energies.append(minimumEnergy)
        else:
            for i in range(self.num_angles):
                minimumEnergy = -FFT_score[i, pred_txy[0], pred_txy[1]].detach().cpu()
                mintxy_energies.append(minimumEnergy)

        xrange = np.arange(0, 2 * np.pi, 2 * np.pi / self.num_angles)
        hardmin_minEnergies = stream_name + '_hardmin' + '_example' + str(plot_count)
        plt.plot(xrange, mintxy_energies)
        plt.title('hardmin')
        plt.savefig('figs/rmsd_and_poses/' + hardmin_minEnergies + '.png')


class EnergyBasedModel(nn.Module):
    def __init__(self, dockingFFT, num_angles=1, step_size=10, sample_steps=10, IP=False, IP_MH=False, FI=False, experiment=None, debug=False):
        super(EnergyBasedModel, self).__init__()
        self.debug = debug
        self.num_angles = num_angles

        self.docker = Docker(dockingFFT, num_angles=self.num_angles, debug=self.debug)
        # self.FIdocker = DockerFI(dockingFFT, num_angles=self.num_angles, debug=self.debug)

        self.sample_steps = sample_steps
        self.step_size = step_size
        self.plot_idx = 0

        self.experiment = experiment
        self.sig_alpha = 0.5

        self.IP = IP
        self.IP_MH = IP_MH
        self.FI = FI

    def forward(self, alpha, receptor, ligand, plot_count=1, stream_name='trainset', plotting=False, training=True):
        if self.IP:
            ## BS model brute force eval
            if self.num_angles > 1:
                ### evaluate with brute force
                alpha = 0
                self.docker.eval()
                lowest_energy, alpha, dr, FFT_score = self.docker(receptor, ligand, alpha, plot_count,
                                                                          stream_name, plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.unsqueeze(0).clone(), FFT_score

            ## train giving the ground truth rotation
            lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                                      plot_count=plot_count, stream_name=stream_name,
                                                                      plotting=plotting)

            return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score

        if self.IP_MH:
            if not training:
                debug = False

                self.docker.eval()
                self.previous = None
                noise_alpha = torch.zeros_like(alpha)
                prob_list = []
                for i in range(self.sample_steps):
                    if i == self.sample_steps - 1:
                        plotting = True
                    else:
                        plotting=False
                    rand_rot = noise_alpha.normal_(0, self.sig_alpha)
                    alpha_out = alpha + rand_rot

                    _, _, dr, FFT_score = self.docker(receptor, ligand, alpha_out,
                                                                              plot_count=plot_count, stream_name=stream_name,
                                                                              plotting=plotting)

                    energy = -(torch.logsumexp(FFT_score, dim=(0, 1)) - torch.log(torch.tensor(100**2)))

                    if self.previous is not None:
                        # print(self.previous)
                        a, b, n = 30, 0.5, 2
                        self.sig_alpha = float(b * torch.exp(-((energy-self.previous) / a) ** n))
                        self.step_size = self.sig_alpha
                        prob = min(torch.exp(-(energy-self.previous)).item(), 1)
                        rand0to1 = torch.rand(1).cuda()
                        prob_list.append(prob)
                        # print('current', energy.item(), 'previous', self.previous.item(), 'alpha_out', alpha_out.item(), 'prev alpha', alpha.item())
                        if energy < self.previous:
                            if debug:
                                print('accept <')
                                print('current', energy.item(), 'previous',self.previous.item(),  'alpha_out', alpha_out.item(), 'prev alpha', alpha.item())
                                print('alpha', alpha_out)
                            self.previous = energy
                            alpha = alpha_out
                        elif energy > self.previous and prob > rand0to1:
                            if debug:
                                print('accept > and prob', prob, ' >', rand0to1.item())
                                print('current', energy.item(), 'previous',self.previous.item(),  'alpha_out', alpha_out.item(), 'prev alpha', alpha.item())
                                print('alpha', alpha_out)
                            self.previous = energy
                            alpha = alpha_out
                        else:
                            if debug:
                                print('reject')
                            alpha_out = alpha.unsqueeze(0)
                    else:
                        self.previous = energy

                if debug:
                    plt.close()
                    xrange = np.arange(0, len(prob_list))
                    # y = prob_list
                    y = sorted(prob_list)[::-1]
                    plt.scatter(xrange, y)
                    plt.show()
                return energy, alpha_out.unsqueeze(0).clone(), dr.clone(), FFT_score

            else:
                ## train giving the ground truth rotation
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                                          plot_count=plot_count, stream_name=stream_name,
                                                                          plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score

        if self.FI:
            pass

            # prob = torch.exp(-beta*(energy-previous))
            # if energy < previous:
            #     accept
            # elif energy > previous and rand0to1 < prob:
            #     accept
            # else:
            #     neg_alpha_out = neg_alpha.unsqueeze(0)

    # def forward(self, pos_alpha, neg_alpha, receptor, ligand, plot_count=1, stream_name='trainset', plotting=False, training=True):
    #     ### evaluate with ld
    #     if not training:
    #         # print('evaluating model')
    #         self.EBMdocker.eval()
    #         noise_alpha = torch.zeros_like(neg_alpha)
    #         neg_alpha.requires_grad_()
    #         langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)
    #         langevin_scheduler = optim.lr_scheduler.ExponentialLR(langevin_opt, gamma=0.8)
    #
    #         for i in range(self.sample_steps):
    #             # if i > 1 and i == self.sample_steps - 1:
    #             #     plotting = True
    #                 # print('\nEnergy', Energy)
    #                 # print('sigma', self.sig_alpha)
    #                 # print(langevin_scheduler.get_last_lr())
    #
    #             langevin_opt.zero_grad()
    #
    #             neg_best_score, _, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)
    #
    #             neg_best_score.backward()
    #             langevin_opt.step()
    #             langevin_scheduler.step()
    #
    #
    #             # print(langevin_scheduler.get_last_lr())
    #
    #             if neg_best_score < 0:
    #                 a = 30
    #                 b = 0.5
    #                 n = 2
    #                 self.sig_alpha = float(b*torch.exp(-(neg_best_score/a)**n))
    #                 self.step_size = self.sig_alpha
    #             # print('\nEnergy', neg_best_score)
    #             # print('sigma', self.sig_alpha)
    #             # if neg_best_score < 0:
    #             #     self.sig_alpha = float(-1/neg_best_score)
    #             #     self.step_size = self.sig_alpha
    #             rand_rot = noise_alpha.normal_(0, self.sig_alpha)
    #             neg_alpha = neg_alpha + rand_rot
    #
    #         return neg_best_score, neg_alpha.clone(), neg_dr.clone(), FFT_score
    #     else:
    #         ## train giving the ground truth rotation
    #         pos_best_score, _, pos_dr, FFT_score = self.EBMdocker(receptor, ligand, pos_alpha,
    #                                                                   plot_count=plot_count, stream_name=stream_name,
    #                                                                   plotting=plotting)
    #
    #     return pos_best_score, pos_alpha.unsqueeze(0).clone(), pos_dr.clone(), FFT_score



    # def forward(self, pos_alpha, neg_alpha, receptor, ligand, plot_count=1, stream_name='trainset', plotting=False):
    #     # self.EBMdocker.eval()
    #
    #     ### evaluate with brute force
    #     if self.num_angles > 1:
    #         neg_free_energy, neg_alpha, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count,
    #                                                                    stream_name, plotting=plotting)
    #         return neg_free_energy, neg_free_energy, neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score
    #
    #     goalE, neg_alpha, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, pos_alpha, plot_count,
    #                                                                stream_name, plotting=plotting)
    #     initial_neg_alpha = neg_alpha
    #     for i in range(self.sample_steps):
    #         moveE, neg_alpha, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count,
    #                                                                    stream_name, plotting=plotting)
    #         accept = min(moveE/goalE, 1)
    #         if accept > 0.95:
    #             accepted_neg_alpha = neg_alpha.unsqueeze(0)
    #         else:
    #             accepted_neg_alpha = initial_neg_alpha.unsqueeze(0)
    #
    #     # self.EBMdocker.train()
    #
    #     return goalE, moveE, accepted_neg_alpha.clone(), neg_dr.clone(), FFT_score

    # def forward(self, pos_alpha, neg_alpha, receptor, ligand, plot_count=1, stream_name='trainset', plotting=False):
    #
    #     if self.num_angles > 1:
    #         ### evaluate with brute force
    #         best_score, neg_alpha, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count,
    #                                                                    stream_name, plotting=plotting, pos=False)
    #         return best_score, best_score, neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score
    #
    #     ## ground truth rotation lowest energy
    #     self.EBMdocker.eval()
    #     pos_best_score, _, pos_dr, pos_FFT_score = self.EBMdocker(receptor, ligand, pos_alpha, pos=True, plot_count=plot_count, stream_name=stream_name,
    #                                                                plotting=plotting)
    #     self.EBMdocker.train()
    #
    #     noise_alpha = torch.zeros_like(neg_alpha)
    #     neg_alpha.requires_grad_()
    #     langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)
    #     # langevin_scheduler = optim.lr_scheduler.ExponentialLR(langevin_opt, gamma=0.8)
    #
    #     FFT_score_list = []
    #     for i in range(self.sample_steps):
    #         if i == self.sample_steps - 1:
    #             plotting = True
    #             # print('\nEnergy', neg_free_energy)
    #             # print('sigma', self.sig_alpha)
    #             # print(langevin_scheduler.get_last_lr())
    #
    #         langevin_opt.zero_grad()
    #
    #         rand_rot = noise_alpha.normal_(0, self.sig_alpha)
    #         neg_alpha_step = neg_alpha + rand_rot
    #
    #         # neg_score_step, _, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha_step, plot_count, stream_name,
    #         #                                                    plotting=plotting)
    #         # print('\nEnergy', neg_free_energy)
    #         # print('sigma', self.sig_alpha)
    #
    #         prob = torch.exp(-beta*(energy-previous))
    #         if energy < previous:
    #             accept
    #         elif energy > previous and rand0to1 < prob:
    #             accept
    #         else:
    #             neg_alpha_out = neg_alpha.unsqueeze(0)
    #
    #         # a = 3
    #         # b = 0.5
    #         # n = 4
    #         # self.sig_alpha = float(b*torch.exp(-(neg_free_energy/a)**n))
    #         # self.step_size = self.sig_alpha
    #
    #         FFT_score_list.append(FFT_score)
    #
    #     if self.FI:
    #         FFT_score = torch.stack((FFT_score_list), dim=0)
    #
    #     pos_best_score, _, pos_dr, FFT_score = self.EBMdocker(receptor, ligand, pos_alpha,
    #                                                               plot_count=plot_count, stream_name=stream_name,
    #                                                               plotting=plotting, pos=False)
    #
    #     return pos_best_score, pos_best_score, pos_alpha.unsqueeze(0).clone(), pos_dr.clone(), FFT_score

    # def forward(self, neg_alpha, receptor, ligand, temperature='cold', plot_count=1, stream_name='trainset', plotting=False):
    #     self.EBMdocker.eval()
    #
    #     if self.num_angles > 1:
    #         ### evaluate with brute force
    #         free_energy, neg_alpha, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)
    #         return free_energy, neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score
    #
    #     noise_alpha = torch.zeros_like(neg_alpha)
    #
    #     neg_alpha.requires_grad_()
    #     langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)
    #     # langevin_scheduler = optim.lr_scheduler.ExponentialLR(langevin_opt, gamma=0.8)
    #
    #     FFT_score_list = []
    #     for i in range(self.sample_steps):
    #         if i > 1 and i == self.sample_steps - 1:
    #             plotting = True
    #             # print('\nEnergy', Energy)
    #             # print('sigma', self.sig_alpha)
    #             # print(langevin_scheduler.get_last_lr())
    #
    #         langevin_opt.zero_grad()
    #
    #         free_energy, _, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)
    #
    #         free_energy.backward(retain_graph=True)
    #         # free_energy.backward()
    #         langevin_opt.step()
    #         # langevin_scheduler.step()
    #
    #         rand_rot = noise_alpha.normal_(0, self.sig_alpha)
    #         neg_alpha = neg_alpha + rand_rot
    #
    #         FFT_score_list.append(FFT_score)
    #
    #     if self.FI:
    #         FFT_score = torch.stack((FFT_score_list), dim=0)
    #
    #     self.EBMdocker.train()
    #
    #     return free_energy, neg_alpha.clone(), neg_dr.clone(), FFT_score

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
