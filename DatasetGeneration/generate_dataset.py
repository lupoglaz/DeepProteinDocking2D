import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists
from DeepProteinDocking2D.DatasetGeneration.ProteinPool import ProteinPool, ParamDistribution
from DeepProteinDocking2D.Utility.torchDockingFFT import TorchDockingFFT
from DeepProteinDocking2D.Utility.utility_functions import UtilityFuncs
from DeepProteinDocking2D.Plotting.plot_FI import FIPlotter
from DeepProteinDocking2D.Tests.check_shape_distributions import ShapeDistributions


def generate_shapes(params, savefile, num_proteins=500):
    pool, stats = ProteinPool.generate(num_proteins=num_proteins, params=params)
    pool.save(savefile)
    return stats


def generate_interactions(receptor, ligand, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
    receptor = torch.tensor(receptor, dtype=torch.float).cuda()
    ligand = torch.tensor(ligand, dtype=torch.float).cuda()
    receptor_stack = FFT.make_boundary(receptor)
    ligand_stack = FFT.make_boundary(ligand)
    fft_score = FFT.dock_global(receptor_stack, ligand_stack, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

    return receptor, ligand, fft_score


def plot_energy_dists(weight_string, train_fft_score_list, test_fft_score_list, trainpool_num_proteins, testpool_num_proteins, show=False):
    plt.close()
    plt.title('Docking energies of all pairs')
    plt.ylabel('Counts')
    plt.xlabel('Energies')
    y1, x1, _ = plt.hist(train_fft_score_list[1], alpha=0.33)
    y2, x2, _ = plt.hist(test_fft_score_list[1], alpha=0.33)
    plt.vlines(docking_decision_threshold, ymin=0, ymax=max(y1.max(), y2.max())+1, linestyles='dashed', label='docking decision threshold', colors='k')
    plt.legend(['docking decision threshold', 'training set', 'testing set'])
    savefile = 'Figs/PairEnergyDistributions/energydistribution_' + weight_string +\
               '_trainpool'+str(trainpool_num_proteins) + 'testpool'+str(testpool_num_proteins) + '.png'
    plt.savefig(savefile)
    if show:
        plt.show()


def plot_accepted_rejected_shapes(receptor, ligand, rot, trans, lowest_energy, fft_score, protein_pool_prefix, plot_count, plot_freq):
    if plot_count % plot_freq == 0:
        plt.close()
        pair = UtilityFuncs().plot_assembly(receptor.cpu(), ligand.cpu(), rot.cpu(), trans.cpu())
        plt.imshow(pair.transpose())
        if lowest_energy < docking_decision_threshold:
            acc_or_rej = 'ACCEPTED'
        else:
            acc_or_rej = 'REJECTED'
        title = acc_or_rej + '_docking_energy' + str(lowest_energy.item())
        plt.title(title)
        plt.savefig('Figs/AcceptRejectExamples/' + title + '.png')
        UtilityFuncs().plot_rotation_energysurface(fft_score, trans,
                                                   stream_name=acc_or_rej + '_datasetgen_' + protein_pool_prefix,
                                                   plot_count=plot_count)


def generate_datasets(protein_pool, num_proteins, weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk):
    data = ProteinPool.load(protein_pool)
    protein_pool_prefix = protein_pool[:-4]

    protein_shapes = data.proteins
    fft_score_list = [[], []]
    docking_set = []
    interaction_set = []
    plot_count = 0
    plot_freq = 100

    translation_space = protein_shapes[0].shape[-1]
    volume = torch.log(360 * torch.tensor(translation_space ** 2))

    filename = log_savepath+'log_rawdata_FI_'+protein_pool_prefix+'.txt'
    with open(filename, 'w') as fout:
        fout.write('F\tF_0\tLabel\n')

    for i in tqdm(range(num_proteins)):
        for j in tqdm(range(i, num_proteins)):
            interaction = None
            plot_count += 1
            receptor, ligand = protein_shapes[i], protein_shapes[j]
            receptor, ligand, fft_score = generate_interactions(receptor, ligand,
                                                        weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)

            rot, trans = FFT.extract_transform(fft_score)
            energies = -fft_score
            lowest_energy = energies[rot.long(), trans[0], trans[1]]

            ## picking docking shapes
            if lowest_energy < docking_decision_threshold:
                docking_set.append([receptor, ligand, rot, trans])

            free_energy = -(torch.logsumexp(-energies, dim=(0, 1, 2)) - volume)

            if free_energy < interaction_decision_threshold:
                interaction = torch.tensor(1)
            if free_energy > interaction_decision_threshold:
                interaction = torch.tensor(0)
            interaction_set.append([receptor, ligand, interaction])
            with open(filename, 'a') as fout:
                fout.write('%f\t%s\t%d\n' % (free_energy.item(), 'NA', interaction.item()))

            fft_score_list[0].append([i, j])
            fft_score_list[1].append(lowest_energy.item())

            if plotting:
                plot_accepted_rejected_shapes(receptor, ligand, rot, trans, lowest_energy, fft_score,
                                              protein_pool_prefix, plot_count, plot_freq)

    return fft_score_list, docking_set, interaction_set


if __name__ == '__main__':
    # Initialize random seeds
    # random_seed = 42
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # random.seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.set_device(0)

    ### Initializations START
    plotting = True
    swap_quadrants = False
    trainset_exists = False
    testset_exists = False
    trainset_pool_stats = None
    testset_pool_stats = None

    log_savepath = 'Log/losses/'
    data_savepath = '../Datasets/'

    normalization = 'ortho'
    FFT = TorchDockingFFT(swap_plot_quadrants=swap_quadrants, normalization=normalization)

    trainpool_num_proteins = 5
    testpool_num_proteins = 5

    validation_set_cutoff = 0.8 ## proportion of training set to keep

    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk = 10, 20, 20, 200
    docking_decision_threshold = -90
    interaction_decision_threshold = -90

    weight_string = str(weight_bound) + ',' + str(weight_crossterm1) + ',' + str(weight_crossterm2) + ',' + str(weight_bulk)

    trainvalidset_protein_pool = 'trainvalidset_protein_pool' + str(trainpool_num_proteins) + '.pkl'
    testset_protein_pool = 'testset_protein_pool' + str(testpool_num_proteins) + '.pkl'
    ### Initializations END

    # DONE
    ###  generate figure with alpha vs numpoints
    ### training set -> center dists with regular tails. testing set -> shifted mean longer tails (grab binomial counts)
    ###  orthogonalization of features plotting

    #### TODO: homodimers no detection threshold, if i==j compare energy to i!=j and normalize
    ### TODO: check monte carlo acceptance rate

    ### Generate training/validation set protein pool
    ## dataset parameters (parameter, probability)
    train_alpha = [(0.80, 1), (0.85, 2), (0.90, 1)]
    train_num_points = [(60, 1), (80, 2), (100, 1)]
    train_params = ParamDistribution(alpha=train_alpha, num_points=train_num_points)

    if exists(trainvalidset_protein_pool):
        trainset_exists = True
        print('\n' + trainvalidset_protein_pool, 'already exists!')
        print('This training/validation protein shape pool will be loaded for dataset generation..')
    else:
        print('\n' + trainvalidset_protein_pool, 'does not exist yet...')
        print('Generating pool of', str(trainpool_num_proteins), 'protein shapes for training/validation set...')
        trainset_pool_stats = generate_shapes(train_params, trainvalidset_protein_pool, trainpool_num_proteins)

    ### Generate testing set protein pool
    ## dataset parameters (parameter, probability)
    test_alpha = [(0.70, 1), (0.80, 4), (0.90, 6), (0.95, 4), (0.98, 1)]
    test_num_points = [(40, 1), (60, 3), (80, 3), (100, 1)]
    test_params = ParamDistribution(alpha=test_alpha, num_points=test_num_points)

    if exists(testset_protein_pool):
        testset_exists = True
        print('\n' + testset_protein_pool, 'already exists!')
        print('This testing protein shape pool will be loaded for dataset generation..')
    else:
        print('\n' + testset_protein_pool, 'does not exist yet...')
        print('Generating pool of', str(testset_protein_pool), 'protein shapes for testing set...')
        testset_pool_stats = generate_shapes(test_params, testset_protein_pool, testpool_num_proteins)

    ### Generate training/validation set
    train_fft_score_list, train_docking_set, train_interaction_set = generate_datasets(
                                                    trainvalidset_protein_pool, trainpool_num_proteins,
                                                    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)
    ### Generate testing set
    test_fft_score_list, test_docking_set, test_interaction_set = generate_datasets(
                                                    testset_protein_pool, testpool_num_proteins,
                                                    weight_bound, weight_crossterm1, weight_crossterm2, weight_bulk)


    ## Slice validation set out for training set
    valid_docking_cutoff_index = int(len(train_docking_set) * validation_set_cutoff)
    valid_docking_set = train_docking_set[valid_docking_cutoff_index:]
    valid_interaction_cutoff_index = int(len(train_interaction_set) * validation_set_cutoff)
    valid_interaction_set = train_interaction_set[valid_interaction_cutoff_index:]

    # print(train_fft_score_list)
    print('Protein Pool:', trainpool_num_proteins)
    print('Docking decision threshold ', docking_decision_threshold)
    print('Interaction decision threshold ', interaction_decision_threshold)

    print('Raw Training set:')
    print('Docking set length', len(train_docking_set))
    print('Interaction set length', len(train_interaction_set))

    print('Raw Validation set:')
    print('Docking set length', len(valid_docking_set))
    print('Interaction set length', len(valid_interaction_set))

    print('Raw Testing set:')
    print('Docking set length', len(test_docking_set))
    print('Interaction set length', len(test_interaction_set))

    ## Write protein pool summary statistics to file
    if not trainset_exists:
        with open(data_savepath + 'protein_trainpool_stats_' + str(trainpool_num_proteins) + 'pool.txt', 'w') as fout:
            fout.write('TRAIN/VALIDATION SET PROTEIN POOL STATS')
            fout.write('\nProtein Pool size='+str(trainpool_num_proteins)+':')
            fout.write('\nTRAIN set params (alpha, number of points):\n'+ str(train_alpha) +'\n'+str(train_num_points))
            fout.write('\nTRAIN set probabilities: '+ '\nalphas:' + str(trainset_pool_stats[0]) +
                       '\nnumber of points' + str(trainset_pool_stats[1]))

    if not testset_exists:
        with open(data_savepath + 'protein_testpool_stats_' + str(testpool_num_proteins) + 'pool.txt', 'w') as fout:
            fout.write('TEST SET PROTEIN POOL STATS')
            fout.write('\nProtein Pool size='+str(testpool_num_proteins)+':')
            fout.write('\n\nTEST set params  (alpha, number of points):\n'+ str(test_alpha) +'\n'+str(test_num_points))
            fout.write('\nTEST set probabilities: '+ '\nalphas:' + str(testset_pool_stats[0]) +
                       '\nnumber of points' + str(testset_pool_stats[1]))

    ## Write dataset summary statistics to file
    with open(data_savepath + 'trainvalid_dataset_stats_' + str(trainpool_num_proteins) + 'pool.txt', 'w') as fout:
        fout.write('TRAIN DATASET STATS')
        fout.write('\nProtein Pool size='+str(trainpool_num_proteins)+':')
        fout.write('\nScoring Weights: '+weight_string)
        fout.write('\nDocking decision threshold ' + str(docking_decision_threshold))
        fout.write('\nInteraction decision threshold ' + str(interaction_decision_threshold))
        fout.write('\n\nRaw Training set:')
        fout.write('\nDocking set length '+str(len(train_docking_set)))
        fout.write('\nInteraction set length '+str(len(train_interaction_set)))
        fout.write('\n\nRaw Validation set:')
        fout.write('\nDocking set length '+str(len(valid_docking_set)))
        fout.write('\nInteraction set length '+str(len(valid_interaction_set)))

    with open(data_savepath + 'testset_dataset_stats_' + str(testpool_num_proteins) + 'pool.txt', 'w') as fout:
        fout.write('TEST DATASET STATS')
        fout.write('\nProtein Pool size=' + str(testpool_num_proteins) + ':')
        fout.write('\nScoring Weights: ' + weight_string)
        fout.write('\nDocking decision threshold ' + str(docking_decision_threshold))
        fout.write('\nInteraction decision threshold ' + str(interaction_decision_threshold))
        fout.write('\n\nRaw Testing set:')
        fout.write('\nDocking set length '+str(len(test_docking_set)))
        fout.write('\nInteraction set length '+str(len(test_interaction_set)))

    ## Save training sets
    docking_train_file = data_savepath + 'docking_train_' + str(trainpool_num_proteins) + 'pool'
    interaction_train_file = data_savepath + 'interaction_train_' + str(trainpool_num_proteins) + 'pool'
    UtilityFuncs().write_pkl(data=train_docking_set, fileprefix=docking_train_file)
    UtilityFuncs().write_pkl(data=train_interaction_set, fileprefix=interaction_train_file)

    ## Save validation sets
    docking_valid_file = data_savepath + 'docking_valid_' + str(trainpool_num_proteins) + 'pool'
    interaction_valid_file = data_savepath + 'interaction_valid_' + str(trainpool_num_proteins) + 'pool'
    UtilityFuncs().write_pkl(data=valid_docking_set, fileprefix=docking_valid_file)
    UtilityFuncs().write_pkl(data=valid_interaction_set, fileprefix=interaction_valid_file)

    ## Save testing sets
    docking_test_file = data_savepath + 'docking_test_' + str(testpool_num_proteins) + 'pool'
    interaction_test_file = data_savepath + 'interaction_test_' + str(testpool_num_proteins) + 'pool'
    UtilityFuncs().write_pkl(data=test_docking_set, fileprefix=docking_test_file)
    UtilityFuncs().write_pkl(data=test_interaction_set, fileprefix=interaction_test_file)

    if plotting:
        ## Dataset shape pair docking energies distributions
        plot_energy_dists(weight_string, train_fft_score_list, test_fft_score_list, trainpool_num_proteins, testpool_num_proteins, show=True)

        ## Dataset free energy distributions
        ## Plot interaction training/validation set
        training_filename = log_savepath+'log_rawdata_FI_'+trainvalidset_protein_pool[:-4]+'.txt'
        FIPlotter(trainvalidset_protein_pool[:-4]).plot_deltaF_distribution(filename=training_filename, binwidth=1, show=True)

        ## Plot interaction testing set
        testing_filename = log_savepath+'log_rawdata_FI_'+testset_protein_pool[:-4]+'.txt'
        FIPlotter(testset_protein_pool[:-4]).plot_deltaF_distribution(filename=testing_filename, binwidth=1, show=True)

        ## Plot protein pool distribution summary
        ShapeDistributions(trainvalidset_protein_pool, 'trainset', show=True).plot_shapes_and_params()
        ShapeDistributions(testset_protein_pool, 'testset', show=True).plot_shapes_and_params()
