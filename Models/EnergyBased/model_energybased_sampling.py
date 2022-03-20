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
        self.docker = BruteForceDocking(dim=100, num_angles=self.num_angles, debug=debug)
        self.dockingFFT = dockingFFT
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, receptor, ligand, rotation, plot_count=1, stream_name='trainset', plotting=False):
        if 'trainset' not in stream_name:
            training = False
        else: training = True

        FFT_score = self.docker.forward(receptor, ligand, angle=rotation, plotting=plotting, training=training, plot_count=plot_count, stream_name=stream_name)
        E_softmax = self.softmax(FFT_score).squeeze() #* FFT_score

        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
            deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)

            if self.num_angles == 1:
                # best_score = FFT_score[pred_txy[0], pred_txy[1]]
                best_score = E_softmax[pred_txy[0], pred_txy[1]]
            else:
                # best_score = FFT_score[deg_index_rot, pred_txy[0], pred_txy[1]]
                best_score = E_softmax[deg_index_rot, pred_txy[0], pred_txy[1]]

                if plotting and plot_count % 10 == 0:
                    # E_softmax = FFT_score
                    self.plot_rotE_surface(FFT_score, pred_txy, E_softmax, stream_name, plot_count)

        Energy = -best_score

        return Energy, pred_txy, pred_rot, FFT_score

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

    def forward(self, neg_alpha, receptor, ligand, temperature='cold', plot_count=1, stream_name='trainset', plotting=False):
        self.EBMdocker.eval()

        # self.minE = torch.inf

        if self.num_angles > 1:
            ### evaluate with brute force
            _, neg_dr, neg_alpha, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)
            return neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score

        noise_alpha = torch.zeros_like(neg_alpha)

        neg_alpha.requires_grad_()
        langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)
        langevin_scheduler = optim.lr_scheduler.ExponentialLR(langevin_opt, gamma=0.8)
        FFT_score_list = []
        for i in range(self.sample_steps):
            if i > 1 and i == self.sample_steps - 1:
                plotting = True
                # print('\nEnergy', Energy)
                # print('sigma', self.sig_alpha)
                # print(langevin_scheduler.get_last_lr())

            langevin_opt.zero_grad()

            Energy, neg_dr, _, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)

            a = 0.5
            # b = 0.5
            n = 2
            self.sig_alpha = float(a*torch.exp(-(Energy/a)**n))

            Energy.backward()
            langevin_opt.step()
            langevin_scheduler.step()

            rand_rot = noise_alpha.normal_(0, self.sig_alpha)
            neg_alpha = neg_alpha + rand_rot
            # print(rand_rot)
            # neg_alpha = neg_alpha + rand_rot.data.clamp(-0.05, 0.05)

            # if Energy < self.minE:
            #     self.minE = Energy
            #     FFT_score_best = FFT_score
            FFT_score_list.append(FFT_score)

        if self.FI:
            FFT_score = torch.stack((FFT_score_list), dim=0)

        self.EBMdocker.train()

        return neg_alpha.clone(), neg_dr.clone(), FFT_score#_best

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
