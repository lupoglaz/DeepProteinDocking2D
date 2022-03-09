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
        # E_softmax = (self.softmax(FFT_score) * FFT_score)

        with torch.no_grad():
            pred_rot, pred_txy = self.dockingFFT.extract_transform(FFT_score)
            deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % self.num_angles).type(torch.long)
            E_softmax = (self.softmax(FFT_score) * FFT_score)

            if self.num_angles == 1:
                best_score = FFT_score[pred_txy[0], pred_txy[1]]
            else:
                best_score = FFT_score[deg_index_rot, pred_txy[0], pred_txy[1]]
                if plotting and plot_count % 10 == 0:
                    self.plot_rotE_surface(FFT_score, pred_txy, E_softmax, stream_name, plot_count)

        minE = -best_score

        return minE, pred_txy, pred_rot, FFT_score
        # return minE, pred_txy, pred_rot, E_softmax

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
    def __init__(self, dockingFFT, num_angles=1, device='cuda', num_samples=1, weight=1.0, step_size=0.1, sample_steps=1, experiment=None, debug=False):
        super(EnergyBasedModel, self).__init__()
        self.debug = debug
        self.num_angles = num_angles

        self.EBMdocker = DockerEBM(dockingFFT, num_angles=self.num_angles, debug=self.debug)

        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.weight = weight
        self.step_size = step_size
        self.device = device

        self.plot_idx = 0

        self.experiment = experiment

    def forward(self, neg_alpha, neg_dr, receptor, ligand, temperature='cold', plot_count=1, stream_name='trainset', plotting=False):
        # TODO: plot FFTscores and softmax(FFTscores)

        if self.num_angles > 1:
            # if self.sample_steps == 0:
            # print('evaluating with brute force')
            minE, neg_dr, neg_alpha, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)
            return neg_alpha.unsqueeze(0).clone(), neg_dr.unsqueeze(0).clone(), FFT_score, minE

        noise_alpha = torch.zeros_like(neg_alpha)
        # self.EBMdocker.eval()

        neg_alpha.requires_grad_()
        langevin_opt = optim.SGD([neg_alpha], lr=self.step_size, momentum=0.0)

        if temperature == 'cold':
            # self.sig_dr = 0.05
            # self.sig_alpha = 0.5
            # self.sig_alpha = 6.28
            # self.sig_alpha = 0.5
            # self.sig_alpha = 2
            self.sig_alpha = 0.05


        if temperature == 'hot':
            # self.sig_dr = 0.5
            self.sig_alpha = 5
            # self.sig_alpha = 10

        for i in range(self.sample_steps):
            if i == self.sample_steps - 1:
                plotting = True

            langevin_opt.zero_grad()

            minE, neg_dr, pred_rot, FFT_score = self.EBMdocker(receptor, ligand, neg_alpha, plot_count, stream_name, plotting=plotting)

            minE.mean().backward(retain_graph=True)
            langevin_opt.step()

            neg_alpha = neg_alpha + noise_alpha.normal_(0, self.sig_alpha)
            # neg_alpha.data = neg_alpha.data.clamp_(-np.pi, np.pi)
            # TODO: add if statement to clamp alpha changes per iteration (5 degree?)

        # self.EBMdocker.train()

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
