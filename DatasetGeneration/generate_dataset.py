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


def generate_interactions(receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
    receptor = torch.tensor(receptor, dtype=torch.float).cuda()
    ligand = torch.tensor(ligand, dtype=torch.float).cuda()
    receptor_stack = FFT.make_boundary(receptor)
    ligand_stack = FFT.make_boundary(ligand)
    fft_score = FFT.dock_global(receptor_stack, ligand_stack, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
    rot, trans = FFT.extract_transform(fft_score)
    best_score = fft_score[rot.long(), trans[0].long(), trans[1].long()]

    return receptor, ligand, fft_score, rot, trans, best_score


def generate_datasets(protein_pool, num_proteins, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
    protein_pool_prefix = protein_pool[:-4]
    data = Utility().read_pkl(protein_pool_prefix)
    protein_shapes = data[0]
    fft_score_list = [[], []]
    docking_set = []
    interaction_set = []
    plot_count = 0
    for i in tqdm(range(num_proteins)):
        for j in tqdm(range(i, num_proteins)):
            plot_count += 1
            receptor, ligand = protein_shapes[i], protein_shapes[j]
            receptor, ligand, fft_score, rot, trans, best_score = generate_interactions(
                receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

            if best_score > docking_score_threshold:
                if 'test' in protein_pool_prefix:
                    docking_set.append([receptor, ligand, rot, trans])
                elif i != j:
                    docking_set.append([receptor, ligand, rot, trans])

            if best_score > interaction_score_threshold:
                interaction_set.append([receptor, ligand, 1])

            if best_score < interaction_score_threshold:
                interaction_set.append([receptor, ligand, 0])

            fft_score_list[0].append([i, j])
            fft_score_list[1].append(best_score.item())
            if plotting:
                plt.close()
                pair = Utility().plot_assembly(receptor.cpu(), ligand.cpu(), rot.cpu(), trans.cpu())
                plt.imshow(pair.transpose())
                title = 'docking_index_i'+str(i)+'_j'+str(j)+'_score'+str(best_score.item())
                plt.title(title)
                plt.savefig('Figs/'+title+'.png')
                Utility().plot_rotation_energysurface(fft_score, trans, num_angles=360, stream_name='datasetgeneration', plot_count=plot_count)

    return fft_score_list, docking_set, interaction_set


if __name__ == '__main__':
    ## Initialize random seeds
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    # torch.set_printoptions(precision=3)

    plotting = True
    swap_quadrants = False
    FFT = TorchDockingFFT(swap_plot_quadrants=swap_quadrants, normalization='ortho')

    trainpool_num_proteins = 50
    testpool_num_proteins = trainpool_num_proteins // 2

    trainvalidset_protein_pool = 'trainvalidset_protein_pool' + str(trainpool_num_proteins) + '.pkl'
    testset_protein_pool = 'testset_protein_pool' + str(testpool_num_proteins) + '.pkl'

    if exists(trainvalidset_protein_pool):
        print('\n' + trainvalidset_protein_pool, 'already exists!')
        print('This training/validation protein shape pool will be loaded for dataset generation..')
    else:
        train_params = ParamDistribution(
            alpha=[(0.8, 1), (0.9, 9), (0.95, 4)],
            num_points=[(20, 1), (30, 2), (50, 4), (80, 8), (100, 6)]
        )
        print('\n' + trainvalidset_protein_pool, 'does not exist yet...')
        print('Generating pool of', str(trainpool_num_proteins), 'protein shapes for training/validation set...')
        generate_shapes(train_params, trainvalidset_protein_pool, trainpool_num_proteins)

    if exists(testset_protein_pool):
        print('\n' + testset_protein_pool, 'already exists!')
        print('This testing protein shape pool will be loaded for dataset generation..')

    else:
        test_params = ParamDistribution(
            alpha=[(0.85, 4), (0.91, 4), (0.95, 4), (0.98, 4)],
            num_points=[(60, 1), (70, 2), (80, 4), (90, 8), (100, 6), (110, 6), (120, 6)]
        )
        print('\n' + testset_protein_pool, 'does not exist yet...')
        print('Generating pool of', str(testset_protein_pool), 'protein shapes for testing set...')
        generate_shapes(test_params, testset_protein_pool, testpool_num_proteins)

    docking_score_threshold = 50
    interaction_score_threshold = 0
    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 10, 20, 20, 200

    ### Generate training/validation set
    train_fft_score_list, train_docking_set, train_interaction_set = generate_datasets(trainvalidset_protein_pool, trainpool_num_proteins,
                                                                                       weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
    ### Generate testing set
    test_fft_score_list, test_docking_set, test_interaction_set = generate_datasets(testset_protein_pool, testpool_num_proteins,
                                                                                    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

    if plotting:
        plt.close()
        plt.title('Docking Scores by index')
        plt.ylabel('Scores')
        plt.scatter(np.arange(0, len(train_fft_score_list[1])), train_fft_score_list[1])
        # plt.show()
        plt.savefig('Figs/scoredistribution' + trainvalidset_protein_pool[:-4] + '.png')

        plt.close()
        plt.title('Docking Scores by index')
        plt.ylabel('Scores')
        plt.scatter(np.arange(0, len(test_fft_score_list[1])), test_fft_score_list[1])
        # plt.show()
        plt.savefig('Figs/scoredistribution' + testset_protein_pool[:-4] + '.png')

    # print(train_fft_score_list)
    print('Training set:')
    print('Docking set length', len(train_docking_set))
    print('Interaction set length', len(train_interaction_set))

    print('Testing set:')
    print('Docking set length', len(test_docking_set))
    print('Interaction set length', len(test_interaction_set))

    ## Save training sets
    Utility().write_pkl(data=train_docking_set, fileprefix='docking_training_set_'+str(len(train_docking_set))+'examples')
    Utility().write_pkl(data=train_interaction_set, fileprefix='interaction_training_set_'+str(len(train_interaction_set))+'examples')

    ## Save validation sets
    valid_docking_cutoff_index = int(len(train_docking_set) * 0.8)
    valid_docking_set = train_docking_set[valid_docking_cutoff_index:]
    valid_interaction_cutoff_index = int(len(train_interaction_set) * 0.8)
    valid_interaction_set = train_interaction_set[valid_interaction_cutoff_index:]
    Utility().write_pkl(data=train_docking_set, fileprefix='docking_valid_set_'+str(len(valid_docking_set))+'examples')
    Utility().write_pkl(data=train_interaction_set, fileprefix='interaction_valid_set_'+str(len(valid_interaction_set))+'examples')

    ## Save testiong sets
    Utility().write_pkl(data=train_docking_set, fileprefix='docking_test_set_'+str(len(test_docking_set))+'examples')
    Utility().write_pkl(data=train_interaction_set, fileprefix='interaction_test_set_'+str(len(test_interaction_set))+'examples')
