import scipy.ndimage as ndimage
import _pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from DeepProteinDocking2D.Utility.validation_metrics import RMSD


class UtilityFuncs():
    def __init__(self):
        pass

    def write_pkl(self, data, fileprefix):
        '''
        :param data:
        :param filename:
        '''
        print('\nwriting '+fileprefix+' to .pkl\n')
        with open(fileprefix+'.pkl', 'wb') as fout:
            pkl.dump(data, fout)
        fout.close()

    def read_pkl(self, fileprefix):
        '''
        :param filename:
        :return: data
        '''
        print('\nreading '+fileprefix+'.pkl\n')
        with open(fileprefix+'.pkl', 'rb') as fin:
            data = pkl.load(fin)
        fin.close()
        return data

    def write_txt(self, data, fileprefix):
        '''
        :param data:
        :param filename:
        :return: writes text file with
        '''
        print('\nwriting '+fileprefix+' to .txt\n')
        fout = open(fileprefix+'.txt', 'w')
        for example in data:
            fout.write(str(example)[1:-1] + '\n')
        fout.close()

    def plot_coords(self, ax, poly, plot_alpha=0.25):
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=plot_alpha)

    def plot_multipoly(self, multipolygon):
        plt.close()
        fig, ax = plt.subplots()
        ax.axis('equal')
        for poly in multipolygon:
            self.plot_coords(ax, poly)

    def get_rot_mat(self, theta):
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).cuda()

    def rot_img(self, x, theta, dtype):
        rot_mat = self.get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid)
        return x

    def rotate_gridligand(self, ligand, rotation_angle):
        ligand = ndimage.rotate(ligand, rotation_angle, reshape=False, order=3, mode='nearest', cval=0.0)
        return ligand

    def translate_gridligand(self, ligand, tx, ty):
        ligand = ndimage.shift(ligand, [tx, ty], mode='wrap', order=3, cval=0.0)
        return ligand

    def plot_assembly(self, receptor, ligand, gt_rot, gt_txy, pred_rot=None, pred_txy=None):
        box_size = receptor.shape[-1]
        receptor_copy = receptor * -100
        ligand_copy = ligand * 200

        padding = box_size//2
        if box_size < 100:
            receptor_copy = np.pad(receptor_copy, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
            ligand_copy = np.pad(ligand_copy, ((padding, padding), (padding, padding)), 'constant', constant_values=0)

        inputshapes = receptor_copy + ligand_copy

        gt_rot = (gt_rot * 180.0/np.pi)
        gt_transformlig = self.rotate_gridligand(ligand_copy, gt_rot)
        gt_transformlig = self.translate_gridligand(gt_transformlig, gt_txy[0], gt_txy[1])
        gt_transformlig += receptor_copy

        if pred_txy is not None and pred_rot is not None:
            pred_rot = (pred_rot * 180.0 / np.pi)
            transformligand = self.rotate_gridligand(ligand_copy, pred_rot)
            transformligand = self.translate_gridligand(transformligand, pred_txy[0], pred_txy[1])
            transformligand += receptor_copy

            pair = np.vstack((gt_transformlig, inputshapes, transformligand))
        else:
            pair = np.vstack((gt_transformlig, inputshapes))

        return pair

    def plot_rotation_energysurface(self, fft_score, pred_txy, num_angles=360, stream_name=None, plot_count=0):
        plt.close()
        mintxy_energies = []
        if num_angles == 1:
            minimumEnergy = -fft_score[pred_txy[0], pred_txy[1]].detach().cpu()
            mintxy_energies.append(minimumEnergy)
        else:
            for i in range(num_angles):
                minimumEnergy = -fft_score[i, pred_txy[0], pred_txy[1]].detach().cpu()
                mintxy_energies.append(minimumEnergy)

        xrange = np.arange(0, 2 * np.pi, 2 * np.pi / num_angles)
        hardmin_minEnergies = stream_name + '_energysurface' + '_example' + str(plot_count)
        plt.plot(xrange, mintxy_energies)
        plt.title('Best Scoring Translation Energy Surface')
        plt.ylabel('Energy')
        plt.xlabel('Rotation (rads)')
        plt.savefig('Figs/EnergySurfaces/' + hardmin_minEnergies + '.png')

    def plot_predicted_pose(self, receptor, ligand, gt_rot, gt_txy, pred_rot, pred_txy, plot_count, stream_name):
        plt.close()
        plt.figure(figsize=(8, 8))
        # pred_rot, pred_txy = self.dockingFFT.extract_transform(fft_score)
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('RMSD', rmsd_out.item())

        pair = self.plot_assembly(receptor.squeeze().detach().cpu().numpy(), ligand.squeeze().detach().cpu().numpy(),
                                            pred_rot.detach().cpu().numpy(),
                                            (pred_txy[0].detach().cpu().numpy(), pred_txy[1].detach().cpu().numpy()),
                                            gt_rot.squeeze().detach().cpu().numpy(), gt_txy.squeeze().detach().cpu().numpy())

        plt.imshow(pair.transpose())
        plt.title('Ground truth', loc='left')
        plt.title('Input')
        plt.title('Predicted pose', loc='right')
        plt.text(225, 110, "RMSD = " + str(rmsd_out.item())[:5], backgroundcolor='w')
        plt.grid(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
        plt.savefig('Figs/Features_and_poses/'+stream_name+'_docking_pose_example' + str(plot_count) + '_RMSD' + str(rmsd_out.item())[:4] + '.png')
        # plt.show()

    def plot_features(self, rec_feat, lig_feat, receptor, ligand, scoring_weights, plot_count=0, stream_name='trainset'):
        boundW, crosstermW1, crosstermW2, bulkW = scoring_weights
        if plot_count == 0:
            print('\nLearned scoring coefficients')
            print('bound', str(boundW.item())[:6])
            print('crossterm1', str(crosstermW1.item())[:6])
            print('crossterm2', str(crosstermW2.item())[:6])
            print('bulk', str(bulkW.item())[:6])
        plt.close()
        plt.figure(figsize=(8, 8))
        if rec_feat.shape[-1] < receptor.shape[-1]:
            pad_size = (receptor.shape[-1] - rec_feat.shape[-1]) // 2
            if rec_feat.shape[-1] % 2 == 0:
                rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
                lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            else:
                rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
                                 value=0)
                lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size + 1, pad_size, pad_size + 1]), mode='constant',
                                 value=0)

        pad_size = (receptor.shape[-1]) // 2
        receptor = F.pad(receptor, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        ligand = F.pad(ligand, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        rec_feat = F.pad(rec_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        lig_feat = F.pad(lig_feat, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        rec_plot = np.hstack((receptor.squeeze().t().detach().cpu(),
                              rec_feat[0].squeeze().t().detach().cpu(),
                              rec_feat[1].squeeze().t().detach().cpu()))
        lig_plot = np.hstack((ligand.squeeze().t().detach().cpu(),
                              lig_feat[0].squeeze().t().detach().cpu(),
                              lig_feat[1].squeeze().t().detach().cpu()))

        plt.imshow(np.vstack((rec_plot, lig_plot)), vmin=0, vmax=1)  # plot scale limits
        plt.title('Input', loc='left')
        plt.title('F1_bulk')
        plt.title('F2_bound', loc='right')
        plt.grid(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
        plt.savefig('Figs/Features_and_poses/'+stream_name+'_docking_feats'+'_example' + str(plot_count)+'.png')
        # plt.show()

if __name__ == '__main__':
    print('works')
