import torch
import numpy as np
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
import matplotlib.pyplot as plt


class EBMPlotter:
    def __init__(self, model):
        self.model = model
        self.plot_freq = 100

    def plot_energy_and_pose(self, pos_idx, L_p, L_n, epoch, receptor, ligand, pos_alpha, pos_dr, neg_alpha, neg_dr,
                             filename):
        if pos_idx % self.plot_freq == 0:
            print('PLOTTING LOSS')
            self.plot_IP_energy_loss(L_p.detach().cpu().numpy(), L_n.squeeze().detach().cpu().numpy(), epoch, pos_idx,
                                     filename)
            print('PLOTTING PREDICTION')
            self.plot_pose(receptor, ligand, neg_alpha.squeeze(), neg_dr.squeeze(), 'Pose after LD',
                           filename, pos_idx, epoch,
                           pos_alpha.squeeze().detach().cpu(), pos_dr.squeeze().detach().cpu())

    def plot_feats(self, neg_rec_feat, neg_lig_feat, epoch, pos_idx, filename):
        if pos_idx % self.plot_freq == 0:
            with torch.no_grad():
                neg_rec_feat, neg_lig_feat = self.eigenvec_feats(neg_rec_feat.detach().cpu(),
                                                                 neg_lig_feat.detach().cpu())
                neg_rec_bulk, neg_rec_bound = neg_rec_feat.squeeze()[0, :, :], neg_rec_feat.squeeze()[1, :, :]
                neg_lig_bulk, neg_lig_bound = neg_lig_feat.squeeze()[0, :, :], neg_lig_feat.squeeze()[1, :, :]
                lig_plot = np.hstack((neg_lig_bulk, neg_lig_bound))
                rec_plot = np.hstack((neg_rec_bulk, neg_rec_bound))
                neg_plot = np.vstack((rec_plot, lig_plot))
                plt.imshow(neg_plot)
                plt.colorbar()
                plt.title('Bulk', loc='left')
                plt.title('Example' + str(pos_idx.item()))
                plt.title('Boundary', loc='right')

                plt.savefig(filename)
                plt.close()

    def eigenvec_feats(self, neg_rec_feat, neg_lig_feat):
        A = self.model.scorer[0].weight.view(2, 2).detach().cpu().clone()
        eigvals, V = torch.linalg.eig(A)
        V = V.real
        rv01 = V[0, 0] * neg_rec_feat[:, 0, :, :] + V[1, 0] * neg_rec_feat[:, 1, :, :]
        rv02 = V[0, 1] * neg_rec_feat[:, 0, :, :] + V[1, 1] * neg_rec_feat[:, 1, :, :]
        repr_0 = torch.stack([rv01, rv02], dim=0).unsqueeze(dim=0).detach()
        # print(V)

        rv01 = V[0, 0] * neg_lig_feat[:, 0, :, :] + V[1, 0] * neg_lig_feat[:, 1, :, :]
        rv02 = V[0, 1] * neg_lig_feat[:, 0, :, :] + V[1, 1] * neg_lig_feat[:, 1, :, :]
        repr_1 = torch.stack([rv01, rv02], dim=0).unsqueeze(dim=0).detach()
        # print(V)

        reprs = torch.cat([repr_0, repr_1], dim=0)

        return reprs

    def plot_pose(self, receptor, ligand, rotation, translation, plot_title, filename, pos_idx, epoch, gt_rot=0,
                  gt_txy=(0, 0), pred_interact=None, gt_interact=None):
        if pos_idx % self.plot_freq == 0:
            pair = plot_assembly(receptor.squeeze().detach().cpu().numpy(),
                                 ligand.squeeze().detach().cpu().numpy(),
                                 rotation.detach().cpu().numpy(),
                                 (translation.squeeze()[0].detach().cpu().numpy(),
                                  translation.squeeze()[1].detach().cpu().numpy()),
                                 gt_rot,
                                 gt_txy)

            plt.imshow(pair[:, :].transpose())
            plt.title('EBM Input', loc='left')
            if gt_interact is not None and pred_interact is not None:
                plt.title('Interaction: gt=' + str(gt_interact) + ' pred=' + str(pred_interact)[:3])
            plt.title(plot_title, loc='right')
            plt.suptitle(filename)

            plt.savefig(filename)
            plt.close()

    def plot_IP_energy_loss(self, L_p, L_n, epoch, pos_idx, filename):
        print('L_p, L_n', L_p, L_n)
        f, ax = plt.subplots(figsize=(6, 6))

        axes_lim = (-0.25, 0.25)
        ax.scatter(L_n, L_p, c=".3")
        ax.plot(axes_lim, axes_lim, ls="--", c=".3")
        ax.set(xlim=axes_lim, ylim=axes_lim)
        ax.set_ylabel('L_p')
        ax.set_xlabel('L_n two temp simulation ')
        plt.quiver([0], [0], [L_n], [L_p], angles='xy', scale_units='xy', scale=1)
        plt.quiver([0], [L_p], color=['r'], angles='xy', scale_units='xy', scale=1)
        plt.quiver([L_n], [0], color=['b'], angles='xy', scale_units='xy', scale=1)
        plt.title(
            'IP Loss: Difference in L_p and L_n\n' + 'epoch ' + str(epoch) + ' example number' + str(pos_idx.item()))

        plt.savefig(filename)
        plt.close()
