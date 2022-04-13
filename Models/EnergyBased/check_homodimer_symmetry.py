import torch
from DeepProteinDocking2D.torchDataset import get_docking_stream
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from DeepProteinDocking2D.Models.BruteForce.TorchDockingFFT import TorchDockingFFT

if __name__ == "__main__":
    testset = 'toy_concave_data/docking_data_test'
    max_size = 542
    train_stream = get_docking_stream(testset + '.pkl', batch_size=1, max_size=max_size)

    swap_quadrants = False
    if swap_quadrants:
        FFT = TorchDockingFFT(swap_plot_quadrants=True)
    else:
        FFT = TorchDockingFFT(swap_plot_quadrants=False)

    homodimer_Elist = []
    heterodimer_Elist = []
    Energy_list = []
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

        # rec_vol = torch.sum(receptor)
        # lig_vol = torch.sum(ligand)

        receptor_stack = FFT.make_boundary(receptor)
        ligand_stack = FFT.make_boundary(ligand)
        FFT_score = FFT.dock_global(receptor_stack, ligand_stack,
                                    # weight_bound=3.0, weight_crossterm1=-0.3, weight_crossterm2=-0.3, weight_bulk=2.8)
                                    weight_bound = 3.0, weight_crossterm1 = -0.3, weight_crossterm2 = -0.3, weight_bulk = 30)
                                    # weight_bound=2.8, weight_crossterm1=-0.3, weight_crossterm2=-0.3, weight_bulk=3.0)

        pred_rot, pred_txy = FFT.extract_transform(FFT_score)
        deg_index_rot = (((pred_rot * 180.0 / np.pi) + 180.0) % 360).type(torch.long)
        Energy = -FFT_score[deg_index_rot, pred_txy[0], pred_txy[1]].detach().cpu()\
                 /100
        Energy_list.append(float(Energy.detach().cpu()))

        if torch.sum(receptor - ligand) == 0:
            print('homodimer found')
            homodimer_Elist.append(Energy)
            # plt.imshow(np.hstack((receptor.detach().cpu(), ligand.detach().cpu())))
            # plt.show()
        else:
            print('heterodimer...')
            heterodimer_Elist.append(Energy)
            # plt.imshow(np.hstack((receptor.detach().cpu(), ligand.detach().cpu())))
            # plt.show()


    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.title('homodimer and heterodimer best scoring poses')
    binwidth = 1
    bins = np.arange(min(Energy_list), max(Energy_list) + binwidth, binwidth)

    plt.hist([homodimer_Elist], label=['homodimer'], bins=bins, rwidth=binwidth, alpha=0.33)
    plt.hist([heterodimer_Elist], label=['heterodimer'], bins=bins, rwidth=binwidth, alpha=0.33)
    plt.legend(['homodimers', 'heterodimers'])

    plt.show()

    print('number of homodimers', len(homodimer_Elist))
    print('number of heterodimers', len(heterodimer_Elist))
