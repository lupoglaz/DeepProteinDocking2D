import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from DeepProteinDocking2D.Models.BruteForce.validation_metrics import RMSD
from DeepProteinDocking2D.Models.BruteForce.utility_functions import plot_assembly
import numpy as np


class TorchDockingFFT:
    def __init__(self, dim=100, num_angles=360, angle=None, swap_plot_quadrants=False, debug=False):
        self.debug = debug
        self.swap_plot_quadrants = swap_plot_quadrants
        self.dim = dim
        self.num_angles = num_angles
        if self.num_angles == 1 and angle:
            self.angles = angle.squeeze().unsqueeze(0).cuda()
        else:
            self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=self.num_angles)).cuda()

    def encode_transform(self, gt_rot, gt_txy):
        empty_3D = torch.zeros([self.num_angles, self.dim, self.dim], dtype=torch.double).cuda()
        deg_index_rot = (((gt_rot * 180.0/np.pi) + 180.0) % self.num_angles).type(torch.long)
        centered_txy = gt_txy.type(torch.long)

        if self.num_angles == 1:
            # print(empty_3D.shape)
            # print(centered_txy.shape)
            # centered_txy = centered_txy.squeeze()
            empty_3D[:, centered_txy[0], centered_txy[1]] = 1
        else:
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
        if self.swap_plot_quadrants:
            pred_X = XYind // self.dim - self.dim//2
            pred_Y = XYind % self.dim - self.dim//2
        else:
            pred_X = XYind // self.dim
            pred_Y = XYind % self.dim
        # Just to make translation values look nice in terms of + or - signs
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

    ## Tried some weights suggested by Georgy
    # weight_bound = 1.0, weight_crossterm1 = 1.0, weight_crossterm2 = 1.0, weight_bulk = 1.0,
    # weight_bound = 3.0, weight_crossterm1 = -0.3, weight_crossterm2 = -0.3, weight_bulk = 2.8,
    # weight_bound = 3.0, weight_crossterm1 = -0.3, weight_crossterm2 = -0.3, weight_bulk = 30.0,
    ## Weights learned from model: RECODE_CHECK_BFDOCKING_30epochs DOES NOT WORK WITH RAW BULK BOUNDARY DATASET FEATS
    # weight_bound = 0.7626, weight_crossterm1, 1.0481, weight_crossterm2 = 0.9259, weight_bulk = 0.9861

    def dock_global(self, receptor, ligand, weight_bound=3.0, weight_crossterm1=-0.3, weight_crossterm2=-0.3, weight_bulk=30.0, debug=False):
        initbox_size = receptor.shape[-1]
        # print(receptor.shape)
        pad_size = initbox_size // 2

        f_rec = receptor.unsqueeze(0).repeat(self.num_angles,1,1,1)
        f_lig = ligand.unsqueeze(0).repeat(self.num_angles,1,1,1)
        rot_lig = self.rotate(f_lig, self.angles)

        if initbox_size % 2 == 0:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size, pad_size, pad_size]), mode='constant', value=0)
        else:
            f_rec = F.pad(f_rec, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
            rot_lig = F.pad(rot_lig, pad=([pad_size, pad_size+1, pad_size, pad_size+1]), mode='constant', value=0)
        # print('padded shape', f_rec.shape)

        if debug:
            with torch.no_grad():
                step = 30
                for i in range(rot_lig.shape[0]):
                    if i % step == 0:
                        plt.title('Torch '+str(i)+' degree rotation')
                        plt.imshow(rot_lig[i,0,:,:].detach().cpu())
                        plt.show()

        score = self.CE_dock_translations(f_rec, rot_lig, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
        # score = SwapQuadrants2DFunction.apply(score)

        return score

    def CE_dock_translations(self, receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
        num_feats_per_shape = 2
        receptor_bulk, receptor_bound = torch.chunk(receptor, chunks=num_feats_per_shape, dim=1)
        ligand_bulk, ligand_bound = torch.chunk(ligand, chunks=num_feats_per_shape, dim=1)
        receptor_bulk = receptor_bulk.squeeze()
        receptor_bound = receptor_bound.squeeze()
        ligand_bulk = ligand_bulk.squeeze()
        ligand_bound = ligand_bound.squeeze()
        # print(receptor_bulk.shape)

        # Bulk score
        cplx_rec = torch.fft.rfft2(receptor_bulk, dim=(-2, -1))
        cplx_lig = torch.fft.rfft2(ligand_bulk, dim=(-2, -1))
        trans_bulk = torch.fft.irfft2(cplx_rec * torch.conj(cplx_lig), dim=(-2, -1))
        # print(cplx_lig.shape, cplx_rec.shape)
        # print(cconv.shape)
        # print(trans_bulk.shape)

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

        if self.swap_plot_quadrants:
            return self.swap_quadrants(score)
        else:
            return score

    def check_FFT_predictions(self, FFT_score, receptor, ligand, gt_txy, gt_rot):
        print('\n'+'*'*50)

        pred_rot, pred_txy = self.extract_transform(FFT_score)
        rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
        print('extracted predicted indices', pred_rot, pred_txy)
        print('gt indices', gt_rot, gt_txy)
        print('RMSD', rmsd_out.item())
        print()

        if self.num_angles == 1:
            plt.imshow(FFT_score.detach().cpu())
            plt.show()
        else:
            plt.imshow(FFT_score[pred_rot.long(), :, :].detach().cpu())
            plt.show()

        pair = plot_assembly(receptor.detach().cpu(), ligand.detach().cpu().numpy(), pred_rot.detach().cpu().numpy(),
                             pred_txy.detach().cpu().numpy(), gt_rot.detach().cpu().numpy(), gt_txy.detach().cpu().numpy())
        plt.imshow(pair.transpose())
        plt.show()

    def swap_quadrants(ctx, input_volume):
        # batch_size = input_volume.size(0)
        # num_features = input_volume.size(1)
        num_features = input_volume.size(0)
        L = input_volume.size(-1)
        L2 = int(L / 2)
        # output_volume = torch.zeros(batch_size, num_features, L, L, device=input_volume.device, dtype=input_volume.dtype)
        output_volume = torch.zeros(num_features, L, L, device=input_volume.device, dtype=input_volume.dtype)

        output_volume[:, :L2, :L2] = input_volume[:, L2:L, L2:L]
        output_volume[:, L2:L, L2:L] = input_volume[:, :L2, :L2]

        output_volume[:, L2:L, :L2] = input_volume[:, :L2, L2:L]
        output_volume[:, :L2, L2:L] = input_volume[:, L2:L, :L2]

        output_volume[:, L2:L, L2:L] = input_volume[:, :L2, :L2]
        output_volume[:, :L2, :L2] = input_volume[:, L2:L, L2:L]

        return output_volume


if __name__ == '__main__':

    from DeepProteinDocking2D.torchDataset import get_docking_stream
    from tqdm import tqdm

    trainset = 'toy_concave_data/docking_data_test'

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

        num_angles = 360
        # num_angles = 1
        if num_angles == 1:
            FFT = TorchDockingFFT(swap_plot_quadrants=False, num_angles=1, angle=torch.ones(1)*np.pi)
        else:
            FFT = TorchDockingFFT(swap_plot_quadrants=True)

        receptor_stack = FFT.make_boundary(receptor)
        ligand_stack = FFT.make_boundary(ligand)
        FFT_score = FFT.dock_global(receptor_stack, ligand_stack, debug=False)

        FFT.check_FFT_predictions(FFT_score, receptor, ligand, gt_txy, gt_rot)

        # FFT_score = TorchDockingFFT().dock_global(receptor_stack, ligand_stack, weight_bound=-0.3, weight_crossterm1=0.55, weight_crossterm2=0.55, weight_bulk=3.0, debug=False)
        #
        #
        # TorchDockingFFT().check_FFT_predictions(-FFT_score, receptor, ligand, gt_txy, gt_rot)
