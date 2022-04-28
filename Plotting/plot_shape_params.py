import matplotlib.pyplot as plt
import numpy as np
from DeepProteinDocking2D.Utility.utility_functions import UtilityFuncs
import seaborn as sea
sea.set_style("whitegrid")
from matplotlib import gridspec


def plot_fractions(counts, dataname):
    unique = np.unique(counts)
    bins = len(unique)
    y, _, _ = plt.hist(counts, bins=bins)  # just calling this as a shortcut to get counts
    plt.close()
    fracs = y / sum(y)
    print('\n'+dataname+': ')
    print('unique values', unique)
    print('counts', y)
    print('fractions', fracs)

    barwidth = (unique[-1] - unique[-2])/2
    plt.bar(unique, fracs, width=barwidth)
    plt.xticks(unique)
    plt.ylabel('fraction of dataset')
    plt.xlabel(dataname)
    # plt.savefig(dataname+'_barplots.png')
    plt.close()

    return unique, fracs, barwidth


def plot_shape_and_params(protein_pool, datasetname):
    protein_pool_prefix = protein_pool[:-4]
    data = UtilityFuncs().read_pkl(protein_pool_prefix)
    protein_shapes = data[0]
    shape_params = data[1]

    alpha_counts = []
    numpoints_counts = []
    params_list = []

    for dict in shape_params:
        items = list(dict.items())
        alpha = items[0][1]
        numpoints = items[1][1]
        alpha_counts.append(alpha)
        numpoints_counts.append(numpoints)
        params_list.append([alpha, numpoints])

    alpha_unique, alpha_fracs, alpha_barwidth = plot_fractions(counts=alpha_counts, dataname='alphas')
    numpoints_unique, numpoints_fracs, numpoints_barwidth = plot_fractions(counts=numpoints_counts, dataname='number of points')

    shapes_plot, num_rows, num_cols = plot_shape_distributions(protein_shapes, alpha_unique, numpoints_unique, params_list, protein_pool_prefix)

    fig, ax = plt.subplots(2,2, gridspec_kw={'width_ratios': [2, 1],'height_ratios': [1, 1]})
    ax0 = ax[0,0]
    axoff = ax[0,1]
    axoff.set_axis_off()
    ax1 = ax[1,0]
    ax2 = ax[1,1]

    ax0.bar(numpoints_unique, numpoints_fracs, numpoints_barwidth)
    ax0.grid(False)
    plt.setp(ax0, ylabel='fraction')
    plt.setp(ax0, xlabel='number of points')

    ax1.imshow(shapes_plot)
    ax1.grid(False)
    ax1.set_axis_off()

    ax2.barh(alpha_unique, alpha_fracs, alpha_barwidth)
    ax2.set_yticks(alpha_unique)
    ax2.grid(False)
    plt.setp(ax2, xlabel='fraction')
    plt.setp(ax2, ylabel='alphas')

    # ax1.get_shared_x_axes().join(ax0, ax2)
    # ax1.sharex(ax0)
    # ax1.sharey(ax2)
    plt.tight_layout()
    # plt.subplot_tool()

    plt.savefig(datasetname+str(len(protein_shapes))+'pool_combined_shapes_params')
    plt.show()

    # gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    # ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[2])
    # ax2 = plt.subplot(gs[3])
    #
    # ax0.bar(numpoints_unique, numpoints_fracs, numpoints_barwidth)
    # ax0.grid(False)
    # xaxis = plt.setp(ax0.get_xticklabels(), visible=False)
    #
    # ax1.imshow(shapes_plot)
    # ax1.set_axis_off()
    # ax1.grid(False)
    #
    # ax2.barh(alpha_unique, alpha_fracs, alpha_barwidth)
    # ax2.set_yticks(alpha_unique)
    # ax2.grid(False)
    #
    # plt.tight_layout()
    # plt.savefig('combined_shapes_params')
    # plt.show()


def plot_shape_distributions(protein_shapes, alpha_unique, numpoints_unique, params_list, protein_pool_prefix):
    alphas_list = list(alpha_unique)
    numpoints_list = list(numpoints_unique)

    combination_list = [[i,j] for i in alphas_list for j in numpoints_list]

    indices = []
    combination_list_copy = combination_list[:]

    found_list = []
    for i in combination_list_copy:
        for j in range(len(params_list)):
            if i == params_list[j] and i not in found_list:
                # print(i, params_list[j])
                found_list.append(i)
                indices.append(j)

    # print(combination_list)
    # print(combination_list_copy)

    num_rows = len(alphas_list)
    num_cols = len(numpoints_list)

    print(indices)

    plot_ranges = []
    for i in range(num_rows):
        cur_range = [*range(i*num_cols, (i+1)*num_cols)]
        plot_ranges.append(cur_range)

    # print(plot_ranges)

    plot_rows = []
    for i in plot_ranges:
        cur_indices = np.array(indices)[i]
        cur_row = np.array(protein_shapes)[cur_indices]
        plot_rows.append(np.hstack(cur_row))

    shapes_plot = np.vstack(plot_rows[::-1])

    plt.imshow(shapes_plot)
    plt.grid(False)
    # plt.savefig('shape_distribution')
    plt.close()

    return shapes_plot, num_rows, num_cols


if __name__ == "__main__":

    data_path = '../DatasetGeneration/'

    num_proteins = 200
    trainvalidset_protein_pool = data_path+'trainvalidset_protein_pool' + str(num_proteins) + '.pkl'

    plot_shape_and_params(trainvalidset_protein_pool, 'trainset')

    num_proteins = 100
    testset_protein_pool = data_path+'testset_protein_pool' + str(num_proteins) + '.pkl'

    plot_shape_and_params(testset_protein_pool, 'testset')
