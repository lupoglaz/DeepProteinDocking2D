import torch
from torch.nn import functional as F

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from DeepProteinDocking2D.Models.BruteForce.utility_functions import read_pkl
from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
import numpy as np

class TorchDockingFFT:
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

    def dock_global(self, receptor, ligand, weight_bound=3.0, weight_crossterm1=-0.3, weight_crossterm2=-0.3, weight_bulk=2.8, debug=False):
        initbox_size = receptor.shape[-1]
        # print(receptor.shape)
        pad_size = initbox_size // 2

        f_rec = receptor.unsqueeze(0).repeat(self.num_angles,1,1,1)
        f_lig = ligand.unsqueeze(0).repeat(self.num_angles,1,1,1)
        rot_lig = TorchDockingFFT().rotate(f_lig, self.angles)

        if initbox_size % 2 == 0:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        else:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
        # print('padded shape', f_rec.shape)

        # f_rec = receptor.unsqueeze(0).repeat(self.num_angles,1,1,1)
        # f_lig = ligand.unsqueeze(0).repeat(self.num_angles,1,1,1)
        # rot_lig = TorchDockingFilter().rotate(f_lig, self.angles)

        if debug:
            with torch.no_grad():
                step = 30
                for i in range(rot_lig.shape[0]):
                    if i % step == 0:
                        plt.title('Torch '+str(i)+' degree rotation')
                        plt.imshow(rot_lig[i,0,:,:].detach().cpu())
                        plt.show()

        score = TorchDockingFFT().CE_dock_translations(f_rec, rot_lig, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

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

        signal_dim = 2
        # Bulk score
        cplx_rec = torch.rfft(receptor_bulk, signal_ndim=signal_dim)
        cplx_lig = torch.rfft(ligand_bulk, signal_ndim=signal_dim)
        re = cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 0] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 1]
        im = -cplx_rec[:, :, :, 0] * cplx_lig[:, :, :, 1] + cplx_rec[:, :, :, 1] * cplx_lig[:, :, :, 0]
        cconv = torch.stack([re, im], dim=3)
        trans_bulk = torch.irfft(cconv, signal_dim, signal_sizes=(box_size, box_size))

        # print(cplx_rec.shape, cplx_lig.shape)

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

        ## cross-term score maximizing
        score = weight_bound * -trans_bound + weight_crossterm1 * trans_bulk_bound + weight_crossterm2 * trans_bound_bulk + weight_bulk * trans_bulk

        # print(score.shape)

        return score

    @staticmethod
    def check_FFT_predictions(FFT_score, receptor, ligand, gt_txy, gt_rot):
        print('*'*50)

        gt_rot = torch.tensor(gt_rot, dtype=torch.float).cuda()
        gt_txy = torch.tensor(gt_txy, dtype=torch.float).cuda()
        # print(gt_rot)

        pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        print('RMSD', rmsd_out.item())
        print()

        pair = plot_assembly(receptor.detach().cpu(), ligand.detach().cpu().numpy(), pred_rot.detach().cpu().numpy(),
                             pred_txy.detach().cpu().numpy(), gt_rot.detach().cpu().numpy(), gt_txy.detach().cpu().numpy())
        plt.imshow(pair.transpose())
        plt.show()

if __name__ == '__main__':

    from DeepProteinDocking2D.torchDataset import get_docking_stream
    from tqdm import tqdm

    trainset = 'toy_concave_data/docking_data_train'

    train_stream = get_docking_stream(trainset + '.pkl', batch_size=1)

    for data in tqdm(train_stream):
        receptor, ligand, gt_txy, gt_rot, _ = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()
        gt_txy = gt_txy.squeeze()
        gt_rot = gt_rot.squeeze()

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_rot = gt_rot.to(device='cuda', dtype=torch.float)
        gt_txy = gt_txy.to(device='cuda', dtype=torch.float)

        receptor_stack = TorchDockingFFT().make_boundary(receptor)
        ligand_stack = TorchDockingFFT().make_boundary(ligand)
        FFT_score = TorchDockingFFT().dock_global(receptor_stack, ligand_stack, debug=False)

        TorchDockingFFT().check_FFT_predictions(FFT_score, receptor, ligand, gt_txy, gt_rot)
