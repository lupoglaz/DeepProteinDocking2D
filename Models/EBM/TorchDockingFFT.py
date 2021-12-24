import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from DeepProteinDocking2D.Models.EBM.utility_functions import read_pkl
from DeepProteinDocking2D.Models.EBM.validation_metrics import RMSD
from DeepProteinDocking2D.Models.EBM.utility_functions import plot_assembly
import numpy as np

class TorchDockingFFT():
    def __init__(self, rotation):
        self.dim = 100
        self.rotation = rotation
        self.num_angles = 1

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
        pred_index = torch.argmax(pred_score)
        pred_rot = ((pred_index / self.dim ** 2) * np.pi / 180.0) - np.pi

        XYind = torch.remainder(pred_index, self.dim ** 2)
        pred_X = XYind // self.dim
        pred_Y = XYind % self.dim

        # Just to make translations look nice
        if pred_X > self.dim//2:
            pred_X = pred_X - self.dim
        if pred_Y > self.dim//2:
            pred_Y = pred_Y - self.dim

        return pred_rot, torch.stack((pred_X, pred_Y), dim=0).to(dtype=torch.float, device='cuda'), pred_index.type(torch.float)

    def rotate(self, repr, angle):
        alpha = angle.detach()
        T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
        T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
        R = torch.stack([T0, T1], dim=1)
        R = R.reshape(self.num_angles,2,3)
        # print('lig', repr.shape)
        # print('rotation', alpha.shape)
        # print('Rot matrix', R.shape)
        # print('Rot mat reshape', R.shape)
        curr_grid = F.affine_grid(R, size=repr.size(), align_corners=True).type(torch.float)
        return F.grid_sample(repr, curr_grid, align_corners=True)

    def dock_global(self, receptor, ligand, weight_bound = 3.0, weight_crossterm1 = -0.3, weight_crossterm2 = -0.3, weight_bulk = 30.0, debug=False):
        print('IN ANGLE',self.rotation)

        initbox_size = receptor.shape[-1]
        # print(receptor.shape)
        pad_size = initbox_size // 2

        f_rec = receptor.unsqueeze(0).repeat(self.num_angles,1,1,1)
        f_lig = ligand.unsqueeze(0).repeat(self.num_angles,1,1,1)

        rot_lig = self.rotate(f_lig, self.rotation)

        if debug:
            plt.close()
            lig_figs = np.hstack((f_lig[0, 0, :, :].detach().cpu(), rot_lig[0, 0, :, :].detach().cpu()))
            plt.imshow(lig_figs)
            plt.title('Torch rotation')
            # plt.imshow(rot_lig[0, 0, :, :].detach().cpu())
            plt.show()

        if initbox_size % 2 == 0:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        else:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
        # print('padded shape', f_rec.shape)

        score = self.CE_dock_translations(f_rec, rot_lig, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
        # score = SwapQuadrants2DFunction.apply(score)

        return score


    def CE_dock_translations(self, receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
        box_size = receptor.shape[-1]

        receptor_bulk, receptor_bound = torch.chunk(receptor, chunks=2, dim=1)
        ligand_bulk, ligand_bound = torch.chunk(ligand, chunks=2, dim=1)
        receptor_bulk = receptor_bulk.squeeze()
        receptor_bound = receptor_bound.squeeze()
        ligand_bulk = ligand_bulk.squeeze()
        ligand_bound = ligand_bound.squeeze()
        # print(receptor_bulk.shape)

        # Bulk score
        cplx_rec = torch.fft.rfft2(receptor_bulk, dim=(-2, -1))
        cplx_lig = torch.fft.rfft2(ligand_bulk, dim=(-2, -1))
        trans_bulk = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1))

        # Boundary score
        cplx_rec = torch.fft.rfft2(receptor_bound, dim=(-2, -1))
        cplx_lig = torch.fft.rfft2(ligand_bound, dim=(-2, -1))
        trans_bound = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1))

        # Boundary - bulk score
        cplx_rec = torch.fft.rfft2(receptor_bound, dim=(-2, -1))
        cplx_lig = torch.fft.rfft2(ligand_bulk, dim=(-2, -1))
        trans_bulk_bound = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1))

        # Bulk - boundary score
        cplx_rec = torch.fft.rfft2(receptor_bulk, dim=(-2, -1))
        cplx_lig = torch.fft.rfft2(ligand_bound, dim=(-2, -1))
        trans_bound_bulk = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1))

        ## cross-term score maximizing
        score = weight_bound * trans_bound + weight_crossterm1 * trans_bulk_bound + weight_crossterm2 * trans_bound_bulk - weight_bulk * trans_bulk

        # print(score.shape)

        return score

        # return torch.stack((trans_bound, trans_bulk_bound, trans_bound_bulk, trans_bulk), dim=0)
