import torch
from DeepProteinDocking2D.Utility.torchDataLoader import get_interaction_stream
from tqdm import tqdm

from DeepProteinDocking2D.Utility.torchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Plotting.plot_FI import FIPlotter

if __name__ == "__main__":
    # trainset = 'toy_concave_data/interaction_data_train'
    # max_size = 100
    # trainset = '../DatasetGeneration/interaction_training_set_4545examples'
    # trainset = '../DatasetGeneration/interaction_training_set_4734examples'
    trainset = 'interaction_train_set100pool'
    # trainset = 'interaction_train_set200pool'

    max_size = None
    train_stream = get_interaction_stream('../Datasets/'+trainset + '.pkl', batch_size=1, max_size=max_size)

    swap_quadrants = False
    normalization = 'ortho'
    if swap_quadrants:
        FFT = TorchDockingFFT(swap_plot_quadrants=True)
    else:
        FFT = TorchDockingFFT(swap_plot_quadrants=False, normalization=normalization)

    volume = torch.log(360 * torch.tensor(50 ** 2))

    filename = 'Log/losses/log_rawdata_FI_'+trainset+'.txt'
    with open(filename, 'w') as fout:
        fout.write('F\tF_0\tLabel\n')

    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 10, 20, 20, 200

    for data in tqdm(train_stream):
        receptor, ligand, gt_interact = data

        receptor = receptor.to(device='cuda', dtype=torch.float).squeeze()
        ligand = ligand.to(device='cuda', dtype=torch.float).squeeze()
        gt_interact = gt_interact.to(device='cuda', dtype=torch.float).squeeze()

        receptor_stack = FFT.make_boundary(receptor)
        ligand_stack = FFT.make_boundary(ligand)

        fft_score = FFT.dock_global(receptor_stack, ligand_stack,
                                    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

        E = -fft_score
        F = -(torch.logsumexp(-E, dim=(0, 1, 2)) - volume)

        with open(filename, 'a') as fout:
            fout.write('%f\t%s\t%d\n' % (F.item(), 'NA', gt_interact.item()))

    FIPlotter(trainset).plot_deltaF_distribution(filename=filename, binwidth=1, show=True)
