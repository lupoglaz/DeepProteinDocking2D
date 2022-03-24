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

class DockerEBM(nn.Module):
    def __init__(self, dockingFFT, num_angles=1, debug=False):
        super(DockerEBM, self).__init__()
        self.num_angles = num_angles
        self.dim = 100
        self.docker = BruteForceDocking(dim=self.dim, num_angles=self.num_angles, debug=debug)
        self.dockingFFT = dockingFFT
        # self.softmax = torch.nn.Softmax(dim=0)
        # self.softmax = torch.nn.Softmax2d(dim=0)

    def forward(self, receptor, ligand, rotation, plot_count=1, stream_name='trainset', plotting=False, pos=False):
        if 'trainset' not in stream_name:
            training = False
        else: training = True

        FFT_score = self.docker.forward(receptor, ligand, angle=rotation, plotting=plotting, training=training, plot_count=plot_count, stream_name=stream_name)

        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
            deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)

            if self.num_angles == 1:
                best_score = FFT_score[pred_txy[0], pred_txy[1]]
                # best_score = FFT_score.mean()
                pass
            else:
                best_score = FFT_score[deg_index_rot, pred_txy[0], pred_txy[1]]
                if plotting and plot_count % 10 == 0:
                    E_softmax = FFT_score
                    self.plot_rotE_surface(FFT_score, pred_txy, E_softmax, stream_name, plot_count)


        if not pos:
            best_score = FFT_score.mean()

        Energy = -best_score

        return Energy, pred_rot, pred_txy, FFT_score

    def plot_rotE_surface(self, FFT_score, pred_txy, E_softmax, stream_name, plot_count):
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        mintxy_energies = []
        mintxy_energies_softmax = []
        for i in range(self.num_angles):
            minimumEnergy = -FFT_score[i, pred_txy[0], pred_txy[1]].detach().cpu()
            mintxy_energies.append(minimumEnergy)
            minimumEnergy_softmax = -E_softmax[i, pred_txy[0], pred_txy[1]].detach().cpu()
            mintxy_energies_softmax.append(minimumEnergy_softmax)
        # print(mintxy_energies_softmax)
        xrange = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
        softmax_hardmax_minEnergies = stream_name + '_softmax_hardmax' + '_example' + str(plot_count)
        ax[0].plot(xrange, mintxy_energies)
        # ax[1].set_title('Hardmax')
        ax[1].plot(xrange, mintxy_energies_softmax)
        ax[1].set_title('Softmax')
        plt.suptitle(softmax_hardmax_minEnergies)
        plt.savefig('figs/rmsd_and_poses/' + softmax_hardmax_minEnergies + '.png')

        # free_energy = -torch.log(torch.exp(FFT_score).mean())
        # free_energy = -(torch.logsumexp(FFT_score, dim=(0, 1)) - torch.log(torch.tensor(self.dim**2)))
        # with torch.no_grad():
        #     pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
        #     best_score = -FFT_score[pred_txy[0], pred_txy[1]]
        #
        #     if self.num_angles > 1:
        #         deg_rot_index = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)
        #         best_score = -FFT_score[deg_rot_index, pred_txy[0], pred_txy[1]]
        #
        #         if plotting and plot_count % 10 == 0:
        #             # free_energy_list = []
        #             lowest_energy_list = []
        #             for i in range(self.num_angles):
        #                 # free_energy = -torch.log(torch.exp(FFT_score[i,:,:]).mean())
        #                 # free_energy = -(torch.logsumexp(FFT_score[i,:,:], dim=(0, 1)) - torch.log(torch.tensor(self.dim**2)))
        #                 pred_rot_slice, pred_txy_slice = self.dockingFFT.extract_transform(FFT_score[i,:,:])
        #                 lowest_energy = -FFT_score[i, pred_txy_slice[0], pred_txy_slice[1]]
        #                 # free_energy_list.append(free_energy)
        #                 lowest_energy_list.append(lowest_energy)
        #
        #             plt.close()
        #             fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        #             xrange = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
        #             # ax[0].plot(xrange, free_energy_list)
        #             ax[1].plot(xrange, lowest_energy_list)
        #             ax[1].set_title('Hardmin')
        #             freeE_hardmax_minEnergies = stream_name + '_softmax_hardmax' + '_example' + str(plot_count)
        #             plt.savefig('figs/rmsd_and_poses/' + freeE_hardmax_minEnergies + '.png')
        #
        # score_out = best_score
        #
        # return best_score, pred_rot, pred_txy, FFT_score


class EnergyBasedModel(nn.Module):
    def __init__(self, dockingFFT, num_angles=1, step_size=10, sample_steps=1, FI=False, experiment=None, debug=False):
        super(EnergyBasedModel, self).__init__()
        self.debug = debug
        self.num_angles = num_angles

        self.EBMdocker = DockerEBM(dockingFFT, num_angles=self.num_angles, debug=self.debug)

        self.sample_steps = sample_steps
        self.step_size = step_size
        self.plot_idx = 0

        self.experiment = experiment
        self.FI = FI
        self.sig_alpha = 0.05

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

    def forward(self, pos_alpha, neg_alpha, receptor, ligand, plot_count=1, stream_name='trainset', plotting=False):

        if self.num_angles > 1:
            ### evaluate with brute force
            best_score, neg_alpha, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count,
                                                                       stream_name, plotting=plotting, pos=False)
            return best_score, best_score, neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score

        ### ground truth rotation lowest energy
        # self.EBMdocker.eval()
        # pos_best_score, _, pos_dr, pos_FFT_score = self.EBMdocker(receptor, ligand, pos_alpha, pos=True, plot_count=plot_count, stream_name=stream_name,
        #                                                            plotting=plotting)
        # self.EBMdocker.train()

        # noise_alpha = torch.zeros_like(neg_alpha)
        # neg_alpha.requires_grad_()
        # langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)
        # # langevin_scheduler = optim.lr_scheduler.ExponentialLR(langevin_opt, gamma=0.8)
        #
        # FFT_score_list = []
        # for i in range(self.sample_steps):
        #     if i == self.sample_steps - 1:
        #         plotting = True
        #         # print('\nEnergy', neg_free_energy)
        #         # print('sigma', self.sig_alpha)
        #         # print(langevin_scheduler.get_last_lr())
        #
        #     langevin_opt.zero_grad()
        #
        #     rand_rot = noise_alpha.normal_(0, self.sig_alpha)
        #     neg_alpha_step = neg_alpha + rand_rot
        #
        #     # neg_score_step, _, neg_dr, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha_step, plot_count, stream_name,
        #     #                                                    plotting=plotting)
        #     # print('\nEnergy', neg_free_energy)
        #     # print('sigma', self.sig_alpha)
        #     #
        #     # accept = min(abs(neg_score_step)/abs(pos_best_score), 1)
        #     # print(accept)
        #     # print(abs(neg_score_step), abs(pos_best_score))
        #
        #     # if accept > 0.5 and torch.randint(0,2, (1,1)).cuda() == 1:
        #     #     print('accept')
        #     #     neg_alpha = neg_alpha_step
        #     #     neg_alpha_out = neg_alpha_step.unsqueeze(0)
        #     #     neg_score_step.backward(retain_graph=True)
        #     #     langevin_opt.step()
        #     # else:
        #     #     neg_alpha_out = neg_alpha.unsqueeze(0)
        #
        #     # langevin_scheduler.step()
        #
        #     # a = 3
        #     # b = 0.5
        #     # n = 4
        #     # self.sig_alpha = float(b*torch.exp(-(neg_free_energy/a)**n))
        #     # self.step_size = self.sig_alpha
        #
        #     FFT_score_list.append(FFT_score)
        #
        # if self.FI:
        #     FFT_score = torch.stack((FFT_score_list), dim=0)

        pos_best_score, _, pos_dr, FFT_score = self.EBMdocker(receptor, ligand, pos_alpha,
                                                                  plot_count=plot_count, stream_name=stream_name,
                                                                  plotting=plotting, pos=False)

        return pos_best_score, pos_best_score, pos_alpha.unsqueeze(0).clone(), pos_dr.clone(), FFT_score

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
