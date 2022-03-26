import torch
from torch import optim
import torch.nn as nn
import numpy as np
from matplotlib import pylab as plt

from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_docking import BruteForceDocking

import sys
sys.path.append('/home/sb1638/')


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
    def __init__(self, dockingFFT, num_angles=1, step_size=10, sample_steps=10, IP=False, IP_MH=False, IP_EBM=False, FI=False, experiment=None, debug=False):
        super(EnergyBasedModel, self).__init__()
        self.debug = debug
        self.num_angles = num_angles

        self.docker = Docker(dockingFFT, num_angles=self.num_angles, debug=self.debug)

        self.sample_steps = sample_steps
        self.step_size = step_size
        self.plot_idx = 0

        self.experiment = experiment
        self.sig_alpha = 0.5

        self.IP = IP
        self.IP_MH = IP_MH
        self.IP_EBM = IP_EBM

        self.FI = FI

    def forward(self, alpha, receptor, ligand, plot_count=1, stream_name='trainset', plotting=False, training=True):
        if self.IP:
            if training:
                ## train giving the ground truth rotation
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                              plot_count=plot_count, stream_name=stream_name,
                                                              plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score
            else:
                ## BS model brute force eval
                alpha = 0
                self.docker.eval()
                lowest_energy, alpha, dr, FFT_score = self.docker(receptor, ligand, alpha, plot_count,
                                                                          stream_name, plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.unsqueeze(0).clone(), FFT_score

        if self.IP_MH:
            if training:
                ## train giving the ground truth rotation
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                                          plot_count=plot_count, stream_name=stream_name,
                                                                          plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score
            else:
                ## MC sampling eval
                self.docker.eval()
                return self.MCsampling(alpha, receptor, ligand, plot_count, stream_name)

        if self.IP_EBM:
            if training:
                ## train giving the ground truth rotation
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha,
                                                                          plot_count=plot_count, stream_name=stream_name,
                                                                          plotting=plotting)

                return lowest_energy, alpha.unsqueeze(0).clone(), dr.clone(), FFT_score
            else:
                ## MC sampling eval
                self.docker.eval()
                return self.langevin(alpha, receptor, ligand, plot_count, stream_name)

        if self.FI:
            if training:
                ## MC sampling for Fact of interaction training
                return self.MCsampling(alpha, receptor, ligand, plot_count, stream_name, debug=False)
            else:
                ### evaluate with brute force
                self.docker.eval()
                lowest_energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha, plot_count,
                                                                           stream_name, plotting=plotting)
                return lowest_energy, alpha.unsqueeze(0).clone(), dr.unsqueeze(0).clone(), FFT_score

    def MCsampling(self, alpha, receptor, ligand, plot_count, stream_name, debug=False):

        self.previous = None
        noise_alpha = torch.zeros_like(alpha)
        prob_list = []
        for i in range(self.sample_steps):
            if i == self.sample_steps - 1:
                plotting = True
            else:
                plotting = False
            rand_rot = noise_alpha.normal_(0, self.sig_alpha)
            alpha_out = alpha + rand_rot

            _, _, dr, FFT_score = self.docker(receptor, ligand, alpha_out,
                                              plot_count=plot_count, stream_name=stream_name,
                                              plotting=plotting)
            energy = -(torch.logsumexp(FFT_score, dim=(0, 1)) - torch.log(torch.tensor(100 ** 2)))

            if self.previous is not None:
                # print(self.previous)
                # a, b, n = 30, 1, 2 # RMSD 10 both, 100steps
                # a, b, n = 20, 1, 2 # RMSD 11, 100steps
                # a, b, n = 10, 1, 2 # RMSD 13, 100steps
                # a, b, n = 10, 1, 2 # RMSD 13, 100steps
                ### no decay, 100 steps RMSD 20.5
                # a, b, n = 30, 1, 2 # 10steps RMSD 30
                # a, b, n = 30, 1, 4 # 100steps RMSD 8.5
                # a, b, n = 20, 1, 4 # 100steps RMSD 9.39
                # a, b, n = 40, 1, 4 # 100steps RMSD 7.39
                # a, b, n = 40, 1, 6 # 100steps RMSD 8.30
                # a, b, n = 40, 2, 4 # 100steps RMSD 7.01
                # a, b, n = 40, 3, 4 # 100steps RMSD 5.55
                # a, b, n = 50, 3, 4 # 100steps RMSD  6.51
                # a, b, n = 30, 3, 4 # 100steps RMSD 8.19
                # a, b, n = 40, 4, 4 # 100steps RMSD 5.01
                # a, b, n = 40, 4, 4 # replicate works 100steps RMSD 5.01
                # a, b, n = 40, 5, 4 # 100steps RMSD 8.22
                # a, b, n = 40, 4, 4 # 50steps RMSD 10.88
                # a, b, n = 40, 4, 4 # 100steps No-lse RMSD 6.37
                # a, b, n = 40, 4, 4  # 10steps RMSD 28.27
                # a, b, n = 40, 4, 4  # 180steps RMSD 5.3
                a, b, n = 40, 4, 4  # 75steps RMSD 8.50
                self.sig_alpha = float(b * torch.exp(-((energy - self.previous) / a) ** n))
                self.step_size = self.sig_alpha
                prob = min(torch.exp(-(energy - self.previous)).item(), 1)
                rand0to1 = torch.rand(1).cuda()
                prob_list.append(prob)
                if energy < self.previous:
                    if debug:
                        print('accept <')
                        print('current', energy.item(), 'previous', self.previous.item(), 'alpha_out', alpha_out.item(),
                              'prev alpha', alpha.item())
                        print('alpha', alpha_out)
                    self.previous = energy
                    alpha = alpha_out
                    dr_out = dr
                    FFT_score_out = FFT_score
                elif energy > self.previous and prob > rand0to1:
                    if debug:
                        print('accept > and prob', prob, ' >', rand0to1.item())
                        print('current', energy.item(), 'previous', self.previous.item(), 'alpha_out', alpha_out.item(),
                              'prev alpha', alpha.item())
                        print('alpha', alpha_out)
                    self.previous = energy
                    alpha = alpha_out
                    dr_out = dr
                    FFT_score_out = FFT_score
                else:
                    if debug:
                        print('reject')
                    alpha_out = alpha.unsqueeze(0)
                    _, _, dr_out, FFT_score_out = self.docker(receptor, ligand, alpha_out,
                                                          plot_count=plot_count, stream_name=stream_name,
                                                          plotting=plotting)
            else:
                self.previous = energy

        if debug:
            plt.close()
            xrange = np.arange(0, len(prob_list))
            # y = prob_list
            y = sorted(prob_list)[::-1]
            plt.scatter(xrange, y)
            plt.show()
        return energy, alpha_out.unsqueeze(0).clone(), dr_out.clone(), FFT_score_out

    def langevin(self, alpha, receptor, ligand, plot_count, stream_name, plotting=False, debug=False):

        noise_alpha = torch.zeros_like(alpha)

        alpha.requires_grad_()
        langevin_opt = optim.SGD([alpha], lr=self.step_size, momentum=0.0)

        for i in range(self.sample_steps):
            if i == self.sample_steps - 1:
                plotting = True
            else:
                plotting = False

            langevin_opt.zero_grad()

            energy, _, dr, FFT_score = self.docker(receptor, ligand, alpha, plot_count, stream_name, plotting=plotting)
            # energy = -(torch.logsumexp(FFT_score, dim=(0, 1)) - torch.log(torch.tensor(100 ** 2)))

            energy.backward()
            langevin_opt.step()
            # a, b, n = 40, 4, 4  # 100steps RMSD 38.2
            # self.sig_alpha = float(b * torch.exp(-(energy / a) ** n))
            # self.step_size = self.sig_alpha
            rand_rot = noise_alpha.normal_(0, self.sig_alpha)
            alpha = alpha + rand_rot

        return energy, alpha.clone(), dr.clone(), FFT_score

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
