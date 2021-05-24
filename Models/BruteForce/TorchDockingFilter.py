import torch
from torch.nn import functional as F

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from DeepProteinDocking.DatasetGeneration.utility_functions import *
from DeepProteinDocking.DatasetGeneration.grid_rmsd import RMSD


class TorchDockingFilter:
    def __init__(self):
        self.dim = 100
        self.num_angles = 360
        self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=self.num_angles)).cuda()

    def encode_transform(self, gt_rot, gt_txy):
        empty_3D = torch.zeros([self.num_angles, self.dim, self.dim], dtype=torch.double).cuda()
        deg_index_rot = (((gt_rot * 180.0/np.pi) + 180.0) % self.num_angles).type(torch.long)
        centered_txy = gt_txy.type(torch.long)

        empty_3D[deg_index_rot, centered_txy[0], centered_txy[1]] = 1
        target_flatindex = torch.argmax(empty_3D.flatten()).cuda()
        # print(gt_rot, gt_txy)
        # print(deg_index_rot, centered_txy)
        # print(ConcaveTrainer().extract_transform(empty_3D))
        return target_flatindex

    def extract_transform(self, pred_score):
        pred_argmax = torch.argmax(pred_score)
        pred_rot = ((pred_argmax / self.dim ** 2) * np.pi / 180.0) - np.pi

        XYind = torch.remainder(pred_argmax, self.dim ** 2)
        pred_X = XYind // self.dim
        pred_Y = XYind % self.dim
        if pred_X > self.dim//2:
            pred_X = pred_X - self.dim
        if pred_Y > self.dim//2:
            pred_Y = pred_Y - self.dim
        return pred_rot, torch.stack((pred_X, pred_Y), dim=0)

    @staticmethod
    def make_boundary(grid_shape):
        epsilon = 1e-5
        sobel_top = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float, requires_grad=True).cuda()
        sobel_left = sobel_top.t()

        feat_top = F.conv2d(grid_shape.unsqueeze(0).unsqueeze(0), weight=sobel_top.unsqueeze(0).unsqueeze(0), padding=1)
        feat_left = F.conv2d(grid_shape.unsqueeze(0).unsqueeze(0), weight=sobel_left.unsqueeze(0).unsqueeze(0), padding=1)

        top = feat_top.squeeze() + epsilon
        right = feat_left.squeeze() + epsilon
        boundary = torch.sqrt(top ** 2 + right ** 2)
        feat_stack = torch.stack([grid_shape, boundary], dim=0)

        return feat_stack

    @staticmethod
    def rotate(repr, angle):
        alpha = angle.detach()
        T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
        T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
        R = torch.stack([T0, T1], dim=1)
        curr_grid = F.affine_grid(R, size=repr.size(), align_corners=True).type(torch.float)
        return F.grid_sample(repr, curr_grid, align_corners=True)

    def dock_global(self, receptor, ligand, weight_bulk=1.0, weight_bound=1.0, weight_crossterm=1.0, debug=False):
        initbox_size = receptor.shape[-1]
        # print(receptor.shape)
        pad_size = initbox_size // 2

        if initbox_size % 2 == 0:
            receptor = F.pad(receptor, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            ligand = F.pad(ligand, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        else:
            receptor = F.pad(receptor, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
            ligand = F.pad(ligand, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
        # print('padded shape', receptor.shape)

        f_rec = receptor.unsqueeze(0).repeat(self.num_angles,1,1,1)
        # print(f_rec.shape)
        f_lig = ligand.unsqueeze(0).repeat(self.num_angles,1,1,1)
        # print(f_lig.shape)
        rot_lig = TorchDockingFilter().rotate(f_lig, self.angles)

        if debug:
            with torch.no_grad():
                step = 30
                for i in range(rot_lig.shape[0]):
                    if i % step == 0:
                        plt.title('Torch '+str(i)+' degree rotation')
                        plt.imshow(rot_lig[i,0,:,:].detach().cpu())
                        plt.show()

        score = TorchDockingFilter().CE_dock_translations(f_rec, rot_lig, weight_bulk, weight_bound, weight_crossterm)

        return score.flatten()


    def CE_dock_translations(self, receptor, ligand, weight_bulk, weight_bound, weight_crossterm):
        box_size = receptor.shape[-1]

        receptor_bulk, receptor_bound = torch.chunk(receptor, chunks=2, dim=1)
        ligand_bulk, ligand_bound = torch.chunk(ligand, chunks=2, dim=1)
        receptor_bulk = receptor_bulk.squeeze()
        receptor_bound = receptor_bound.squeeze()
        ligand_bulk = ligand_bulk.squeeze()
        ligand_bound = ligand_bound.squeeze()
        # print(receptor_bulk.shape)

        signal_dim = 2
        # Bulk score
        cplx_rec = torch.rfft(receptor_bulk, signal_ndim=signal_dim)
        cplx_lig = torch.rfft(ligand_bulk, signal_ndim=signal_dim)
        re = cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 0] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 1]
        im = -cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 1] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 0]
        cconv = torch.stack([re, im], dim=3)
        trans_bulk = torch.irfft(cconv, signal_dim, signal_sizes=(box_size, box_size))

        # Boundary score
        cplx_rec = torch.rfft(receptor_bound, signal_ndim=signal_dim)
        cplx_lig = torch.rfft(ligand_bound, signal_ndim=signal_dim)
        re = cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 0] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 1]
        im = -cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 1] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 0]
        cconv = torch.stack([re, im], dim=3)
        trans_bound = torch.irfft(cconv, signal_dim, signal_sizes=(box_size, box_size))
        # print(trans_bound.shape)

        # Boundary - bulk score
        cplx_rec = torch.rfft(receptor_bound, signal_ndim=signal_dim)
        cplx_lig = torch.rfft(ligand_bulk, signal_ndim=signal_dim)
        re = cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 0] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 1]
        im = -cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 1] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 0]
        cconv = torch.stack([re, im], dim=3)
        trans_bulk_bound = torch.irfft(cconv, signal_dim, signal_sizes=(box_size, box_size))

        # Bulk - boundary score
        cplx_rec = torch.rfft(receptor_bulk, signal_ndim=signal_dim)
        cplx_lig = torch.rfft(ligand_bound, signal_ndim=signal_dim)
        re = cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 0] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 1]
        im = -cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 1] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 0]
        cconv = torch.stack([re, im], dim=3)
        trans_bound_bulk = torch.irfft(cconv, signal_dim, signal_sizes=(box_size, box_size))

        ## cross-term scoring Energy maximizing
        score = weight_bound * trans_bound + weight_crossterm * (trans_bulk_bound + trans_bound_bulk) - weight_bulk * trans_bulk

        # print(score.shape)

        return score

    @staticmethod
    def check_FFT_predictions(pred_score, receptor, ligand, gt_rot, gt_txy):
        print('*'*50)

        gt_rot = torch.tensor(gt_rot, dtype=torch.float).cuda()
        gt_txy = torch.tensor(gt_txy, dtype=torch.float).cuda()
        # print(gt_rot)

        pred_rot, pred_txy = TorchDockingFilter().extract_transform(pred_score)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        print('RMSD', rmsd_out.item())
        print()


if __name__ == '__main__':

    data = read_pkl('toy_concave_data/scoregridsearch_training_numpoints=50_r=15_a=0.9_fmin=0.05_fmax=0.2_boxsize=50_ScoreMin-++_crossterms_datasize=10_txy_rot')
    receptor, ligand, gt_txy, gt_rot  = data[-1]

    init_dim = receptor.shape[-1]
    ### print(receptor.shape)
    if init_dim > 51:
        receptor = receptor[25:75, 25:75]
        ligand = ligand[25:75, 25:75]

    receptor = torch.tensor(receptor, dtype=torch.float).cuda()
    ligand = torch.tensor(ligand, dtype=torch.float).cuda()
    receptor_stack = TorchDockingFilter().make_boundary(receptor)
    ligand_stack = TorchDockingFilter().make_boundary(ligand)
    # pred_score = TorchDockingFilter().dock_global(receptor, ligand)

    TorchDockingFilter().check_FFT_predictions(TorchDockingFilter().dock_global(receptor_stack, ligand_stack, debug=False), receptor, ligand,
                                               gt_rot, gt_txy)


