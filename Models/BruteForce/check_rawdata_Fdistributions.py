import torch
from DeepProteinDocking2D.torchDataset import get_interaction_stream_balanced
from tqdm import tqdm
import numpy as np

from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Models.BruteForce.model_bruteforce_interaction import BruteForceInteraction
from DeepProteinDocking2D.Models.BruteForce.plot_FI_loss import FILossPlotter

if __name__ == "__main__":
    trainset = 'toy_concave_data/interaction_data_train'
    max_size = 100
    train_stream = get_interaction_stream_balanced(trainset + '.pkl', batch_size=1, max_size=max_size)

    swap_quadrants = False
    if swap_quadrants:
        FFT = TorchDockingFFT(swap_plot_quadrants=True)
    else:
        FFT = TorchDockingFFT(swap_plot_quadrants=False)

    FI = BruteForceInteraction()
    volume = torch.log(360 * torch.tensor(100 ** 2))

    filename = 'Log/losses/log_rawdata_FI_Trainset.txt'
    with open(filename, 'w') as fout:
        fout.write('F\tF_0\tLabel\n')

    # F_0 = torch.ones(1)*-90

    for data in tqdm(train_stream):
        receptor, ligand, gt_interact = data

        receptor = receptor.squeeze()
        ligand = ligand.squeeze()

        receptor = receptor.to(device='cuda', dtype=torch.float)
        ligand = ligand.to(device='cuda', dtype=torch.float)
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float)

        receptor_stack = FFT.make_boundary(receptor)
        ligand_stack = FFT.make_boundary(ligand)
        FFT_score = FFT.dock_global(receptor_stack, ligand_stack,
                                    # weight_bound=3.0, weight_crossterm1=-0.3, weight_crossterm2=-0.3, weight_bulk=2.8)
                                    weight_bound = 3.0, weight_crossterm1 = -0.3, weight_crossterm2 = -0.3, weight_bulk = 30)
                                    # weight_bound=2.8, weight_crossterm1=-0.3, weight_crossterm2=-0.3, weight_bulk=3.0)

        E = -FFT_score
        F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - volume)

        with open(filename, 'a') as fout:
            fout.write('%f\t%f\t%d\n' % (F.item(), F.item(), gt_interact.item()))

    FILossPlotter('ManuscriptWeights').plot_deltaF_distribution(filename=filename, binwidth=100, show=True)
    # FILossPlotter('AdjustedHighBulkWeight=30').plot_deltaF_distribution(filename=filename, binwidth=100, show=True)
