import scipy.ndimage as ndimage
import _pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from validation_metrics import RMSD

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

    # def plot_dataset(dataset, setname, dim0, dim1):
    #     fig, ax = plt.subplots(dim0, dim1, figsize=(15, 10))
    #     plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.8, hspace=0.8)
    #     count = 0
    #     plt.suptitle("Ground Truth Transform                Input Shapes                  Predicted Pose")
    #     for assembly in dataset:
    #         j = count % dim0
    #         k = count % dim1
    #         count += 1
    #         receptor, ligand, score, MAE, (rotation, translation), (truerot, truetxy), _ = assembly
    #
    #         scoring = 'Score =' + str(score)[:5] + ' MAErot=' + str(MAE[0])[:5] + ' MAEtrans=' + str(MAE[1])[:5]
    #
    #         pair = plot_assembly(receptor, ligand, rotation, translation, truerot, truetxy)
    #
    #         ax[j,k].imshow(pair.transpose())
    #         ax[j,k].grid(None)
    #         ax[j,k].axis('Off')
    #         ax[j,k].text(5, 5, scoring, bbox={'facecolor': 'white', 'pad': 5}, fontsize=7)
    #
    #     plt.tight_layout()
    #     plt.savefig('figs/'+setname + str(count)+'.png')
    #     plt.close()

if __name__ == '__main__':
    print('works')
