import _pickle as pkl
from pathlib import Path
import random
import argparse
import sys
import torch
from os.path import exists
from DeepProteinDocking2D.DatasetGeneration.ProteinPool import ProteinPool, ParamDistribution
from DeepProteinDocking2D.Utility.torchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Utility.utility_functions import read_pkl, write_pkl, plot_assembly

def generate_shapes(params, savefile, num_proteins=500):
    pool = ProteinPool.generate(num_proteins=num_proteins, params=params)
    pool.save(savefile)


def generate_interactions(docker, savefile):
    pool = ProteinPool.load(savefile)
    pool.get_interactions(docker)
    pool.save(savefile)


# def extract_docking_dataset(docker, interaction_criteria, savefile,
#                             output_files=['DatasetGeneration/docking_data_train_torchv1p10.pkl',
#                                           'DatasetGeneration/docking_data_valid_torchv1p10.pkl'],
#                             max_num_samples=1100):
#     pool = ProteinPool.load(savefile)
#     dataset_all = pool.extract_docking_dataset(docker, interaction_criteria, max_num_samples=max_num_samples)
#     total_len = len(dataset_all)
#     print(f'Total data length {total_len}')
#     if len(output_files) == 2:
#         train_len = int(0.8 * total_len)
#         valid_len = total_len - train_len
#         print(f'Training/validation split length {train_len} / {valid_len}')
#         random.shuffle(dataset_all)
#         with open(output_files[0], 'wb') as fout:
#             pkl.dump(dataset_all[:train_len], fout)
#         with open(output_files[1], 'wb') as fout:
#             pkl.dump(dataset_all[train_len:], fout)
#     elif len(output_files) == 1:
#         with open(output_files[0], 'wb') as fout:
#             pkl.dump(dataset_all, fout)
#
#
# def extract_interaction_dataset(interaction_criteria, savefile,
#                                 output_files=['DatasetGeneration/interaction_data_train_torchv1p10.pkl',
#                                               'DatasetGeneration/interaction_data_valid_torchv1p10.pkl'],
#                                 num_proteins=100):
#     pool = ProteinPool.load(savefile)
#     if len(output_files) == 2:
#         train_len = int(0.8 * num_proteins)
#         valid_len = num_proteins - train_len
#         print(f'Training/validation split length {train_len} / {valid_len}')
#         interactome_train = pool.extract_interactome_dataset(interaction_criteria, ind_range=(0, train_len))
#         interactome_valid = pool.extract_interactome_dataset(interaction_criteria, ind_range=(train_len, num_proteins))
#         with open(output_files[0], 'wb') as fout:
#             pkl.dump(interactome_train, fout)
#         with open(output_files[1], 'wb') as fout:
#             pkl.dump(interactome_valid, fout)
#     elif len(output_files) == 1:
#         interactome_test = pool.extract_interactome_dataset(interaction_criteria, ind_range=(0, num_proteins))
#         with open(output_files[0], 'wb') as fout:
#             pkl.dump(interactome_test, fout)
#
#
# def visualize(docker, savefile):
#     pool = ProteinPool.load(savefile)
#     pool.plot_interaction_dist(perc=90)
#     # pool.plot_interactions(docker, filename='examples.png', num_plots=10)
#     pool.plot_sample_funnels(docker, filename='funnels.png', range=[(-200, -150), (-200, -100), (-100, -20)])
#     pool.plot_params()
#
#
# def generate_set(num_proteins, params, interaction_criteria_dock, interaction_criteria_inter,
#                  docker, savefile, training_set=True, num_docking_samples=1100, num_inter_proteins=100):
#     pass


# Structures
# if exists(savefile):
# 	print(savefile, 'already exists!')
# 	print('this protein shape pool will be loaded for dataset generation..')
# else:
# 	print(savefile, 'does not exist')
# 	print('generating pool of', str(num_proteins), 'protein shapes...')
# 	generate_shapes(params, savefile, num_proteins)
# 	#Interaction
# 	generate_interactions(docker, savefile)

# if training_set:
# 	#Docking dataset
# 	extract_docking_dataset(docker, interaction_criteria_dock, savefile,
# 							output_files=['DatasetGeneration/docking_data_train_torchv1p10.pkl', 'DatasetGeneration/docking_data_valid_torchv1p10.pkl'],
# 							max_num_samples = num_docking_samples)
# 	# #Interaction dataset
# 	extract_interaction_dataset(interaction_criteria_inter, savefile,
# 							output_files=['DatasetGeneration/interaction_data_train_torchv1p10.pkl', 'DatasetGeneration/interaction_data_valid_torchv1p10.pkl'],
# 							num_proteins = num_inter_proteins)
# else:
# 	#Docking dataset
# 	extract_docking_dataset(docker, interaction_criteria_dock, savefile,
# 							output_files=['DatasetGeneration/docking_data_test_torchv1p10.pkl'],
# 							max_num_samples = num_docking_samples)
# 	#Interaction dataset
# 	extract_interaction_dataset(interaction_criteria_inter, savefile,
# 							output_files=['DatasetGeneration/interaction_data_test_torchv1p10.pkl'],
# 							num_proteins = num_inter_proteins)

# def find_best_pose(receptor, ligand):
#     receptor, ligand = protein_shapes[i], protein_shapes[j]
#     receptor = torch.tensor(receptor, dtype=torch.float).cuda()
#     ligand = torch.tensor(ligand, dtype=torch.float).cuda()
#     receptor_stack = FFT.make_boundary(receptor)
#     ligand_stack = FFT.make_boundary(ligand)
#     fft_score = FFT.dock_global(receptor_stack, ligand_stack)
#     rot, trans = FFT.extract_transform(fft_score)
#     best_score = fft_score[rot, trans[0], trans[1]]
#     return best_score

if __name__ == '__main__':
    protein_pool = 'protein_pool.pkl'
    num_proteins = 100
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

    # # Interaction
    # generate_interactions(docker, protein_pool)

    swap_quadrants = True
    FFT = TorchDockingFFT(swap_plot_quadrants=swap_quadrants)

    data = read_pkl('protein_pool')
    protein_shapes = data[0]

    FFT_score_list = []
    for i in range(2):
        for j in range(2):
            receptor, ligand = protein_shapes[i], protein_shapes[j]
            receptor = torch.tensor(receptor, dtype=torch.float).cuda()
            ligand = torch.tensor(ligand, dtype=torch.float).cuda()
            receptor_stack = FFT.make_boundary(receptor)
            ligand_stack = FFT.make_boundary(ligand)
            fft_score = FFT.dock_global(receptor_stack, ligand_stack)
            rot, trans = FFT.extract_transform(fft_score)
            best_score = fft_score[rot.long(), trans[0].long(), trans[1].long()]
            plot_assembly(receptor, ligand, trans, rot)

            FFT_score_list.append([i, j, best_score])

    print(FFT_score_list)

    # receptor_stack = FFT.make_boundary(receptor)
    # ligand_stack = FFT.make_boundary(ligand)
    # fft_score = FFT.dock_global(receptor_stack, ligand_stack)
    #
    # FFT.check_FFT_predictions(fft_score, receptor, ligand, gt_txy, gt_rot)

# parser = argparse.ArgumentParser(description='Generate protein docking dataset')
# parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
# parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')
# parser.add_argument('-debug', action='store_const', const=lambda:'debug', dest='cmd')
# parser.add_argument('-gen', action='store_const', const=lambda:'gen', dest='act')
# parser.add_argument('-vis', action='store_const', const=lambda:'vis', dest='act')
#
# # parser.add_argument('-IP', action='store_const', const=lambda:'IP', dest='act')
# # parser.add_argument('-FI', action='store_const', const=lambda:'FI', dest='act')
#
#
# # parser.add_argument('-a00', default=3.0, type=float)
# # parser.add_argument('-a10', default=-0.3, type=float)
# # parser.add_argument('-a11', default=2.5, type=float)
# # parser.add_argument('-score_cutoff', default=-100, type=float)
# # parser.add_argument('-funnel_gap_cutoff', default=15, type=float)
# # parser.add_argument('-free_energy_cutoff', default=-100, type=float)
#
# parser.add_argument('-a00', default=3.0, type=float)
# parser.add_argument('-a10', default=-0.3, type=float)
# parser.add_argument('-a11', default=2.8, type=float)
# parser.add_argument('-score_cutoff', default=-10, type=float)
# parser.add_argument('-funnel_gap_cutoff', default=1, type=float)
# parser.add_argument('-free_energy_cutoff', default=-10, type=float)
#
# args = parser.parse_args()
#
# inter_dock = InteractionCriteriaGandGap(free_energy_cutoff=args.free_energy_cutoff, funnel_gap_cutoff=args.funnel_gap_cutoff)
# inter_inter = InteractionCriteriaG(free_energy_cutoff=args.free_energy_cutoff)
# docker = DockerGPU(a00=args.a00, a10=args.a10, a11=args.a11)
#
# if args.cmd is None:
# 	parser.print_help()
# 	sys.exit()
#
# elif args.cmd() == 'train':
# 	params = ParamDistribution(
# 		alpha = [(0.8, 1), (0.9, 9), (0.95, 4)],
# 		num_points = [(20, 1), (30, 2), (50, 4), (80, 8), (100, 6)]
# 		)
#
# 	savefile = 'DatasetGeneration/Data/protein_pool_big_torchv1p10.pkl'
# 	if args.act() == 'gen':
# 		generate_set(500, params, inter_dock, inter_inter, docker, savefile, training_set=True, num_docking_samples=1100, num_inter_proteins=500)
# 	elif args.act() == 'vis':
# 		visualize(docker, savefile)
# 	else:
# 		print('No action specified')
#
# elif args.cmd() == 'test':
# 	params = ParamDistribution(
# 		alpha = [(0.85, 4), (0.91, 4), (0.95, 4), (0.98, 4)],
# 		num_points = [(60, 1), (70, 2), (80, 4), (90, 8), (100, 6), (110, 6), (120, 6)]
# 		)
#
# 	savefile = 'DatasetGeneration/Data/protein_pool_small_torchv1p10.pkl'
# 	if args.act() == 'gen':
# 		generate_set(250, params, inter_dock, inter_inter, docker, savefile, training_set=False, num_docking_samples=1100, num_inter_proteins=250)
# 	elif args.act() == 'vis':
# 		visualize(docker, savefile)
# 	else:
# 		print('No action specified')
#
# elif args.cmd() == 'debug':
# 	params = ParamDistribution(
# 		alpha = [(0.85, 4), (0.91, 4), (0.95, 4), (0.98, 4)],
# 		num_points = [(60, 1), (70, 2), (80, 4), (90, 8), (100, 6), (110, 6), (120, 6)]
# 		)
#
# 	savefile = 'DatasetGeneration/Data/protein_pool_debug_torchv1p10.pkl'
# 	if args.act() == 'gen':
# 		generate_set(20, params, inter_dock, inter_inter, docker, savefile, training_set=False, num_docking_samples=30, num_inter_proteins=20)
# 	elif args.act() == 'vis':
# 		visualize(docker, savefile)
# 	else:
# 		print('No action specified')
