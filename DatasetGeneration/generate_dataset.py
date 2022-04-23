import _pickle as pkl
from pathlib import Path
import argparse
import sys
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from os.path import exists
from DeepProteinDocking2D.DatasetGeneration.ProteinPool import ProteinPool, ParamDistribution
from DeepProteinDocking2D.Utility.torchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Utility.utility_functions import Utility

def generate_shapes(params, savefile, num_proteins=500):
    pool = ProteinPool.generate(num_proteins=num_proteins, params=params)
    pool.save(savefile)


def generate_interactions(docker, savefile):
    pool = ProteinPool.load(savefile)
    pool.get_interactions(docker)
    pool.save(savefile)


if __name__ == '__main__':

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.set_printoptions(precision=3)


    swap_quadrants = False
    # swap_quadrants = True
    FFT = TorchDockingFFT(swap_plot_quadrants=swap_quadrants, normalization='ortho')
    Utility = Utility()
    plotting = True
    num_proteins = 10
    protein_pool = 'protein_pool'+str(num_proteins)+'.pkl'
    params = ParamDistribution(
        alpha=[(0.8, 1), (0.9, 9), (0.95, 4)],
        num_points=[(20, 1), (30, 2), (50, 4), (80, 8), (100, 6)]
    )
    if exists(protein_pool):
        print(protein_pool, 'already exists!')
        print('this protein shape pool will be loaded for dataset generation..')
    else:
        print(protein_pool, 'does not exist')
        print('generating pool of', str(num_proteins), 'protein shapes...')
        generate_shapes(params, protein_pool, num_proteins)



    data = Utility.read_pkl(protein_pool[:-4])
    protein_shapes = data[0]

    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 10, 20, 20, -200
    # weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 1, 2, 2, -20
    docking_score_threshold = 50
    docking_set_length = 50
    interaction_score_threshold = 0
    interaction_set_length = 100
    FFT_score_list = [[], []]
    docking_set = []
    interaction_set = [[],[]]
    plot_count = 0

    interaction_matrix = torch.zeros(num_proteins, num_proteins)

    for i in tqdm(range(num_proteins)):
        for j in tqdm(range(num_proteins)):
            plot_count += 1
            receptor, ligand = protein_shapes[i], protein_shapes[j]
            receptor = torch.tensor(receptor, dtype=torch.float).cuda()
            ligand = torch.tensor(ligand, dtype=torch.float).cuda()
            receptor_stack = FFT.make_boundary(receptor)
            ligand_stack = FFT.make_boundary(ligand)
            fft_score = FFT.dock_global(receptor_stack, ligand_stack, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
            rot, trans = FFT.extract_transform(fft_score)
            best_score = fft_score[rot.long(), trans[0].long(), trans[1].long()]

            if best_score > docking_score_threshold and len(docking_set) < docking_set_length:
                docking_set.append([receptor, ligand, rot, trans])
            if best_score > interaction_score_threshold and len(interaction_set) < interaction_set_length:
                interaction_set[0].append([receptor, ligand])
                interaction_matrix[i, j] = 1
            if best_score < interaction_score_threshold and len(interaction_set) < interaction_set_length:
                interaction_set.append([receptor, ligand])
            interaction_set[1].append(interaction_matrix)

            # fft_score[rot.long(), trans[0].long(), trans[1].long()] -= 100
            # rot2nd, trans2nd = FFT.extract_transform(fft_score)
            # best_score_2nd = fft_score[rot2nd.long(), trans2nd[0].long(), trans2nd[1].long()]
            # FFT_score_list[2].append(best_score_2nd.item())

            FFT_score_list[0].append([i, j])
            FFT_score_list[1].append(best_score.item())
            if plotting:
                pair = Utility.plot_assembly(receptor.cpu(), ligand.cpu(), rot.cpu(), trans.cpu())
                plt.imshow(pair.transpose())
                title = 'docking_index_i'+str(i)+'_j'+str(j)+'_score'+str(best_score.item())
                plt.title(title)
                plt.savefig('Figs/'+title+'.png')
                Utility.plot_rotation_energysurface(fft_score, trans, num_angles=360, stream_name='datasetgeneration', plot_count=plot_count)

    if plotting:
        plt.close()
        plt.title('Docking Scores by index')
        plt.ylabel('Scores')
        plt.scatter(np.arange(0, num_proteins**2), FFT_score_list[1])
        # plt.show()
        if swap_quadrants:
            plt.savefig('Figs/scoredistribution' + protein_pool[:-4] + 'SwapQuadrants.png')
        else:
            plt.savefig('Figs/scoredistribution' + protein_pool[:-4] + '.png')

    print(FFT_score_list)
    print('Docking set length', len(docking_set))
    print('Interaction set length', len(interaction_set))

    Utility.write_pkl(data=docking_set, fileprefix='docking_set_'+str(len(docking_set))+'examples')
    Utility.write_pkl(data=interaction_set, fileprefix='docking_set_'+str(len(interaction_set))+'examples')
