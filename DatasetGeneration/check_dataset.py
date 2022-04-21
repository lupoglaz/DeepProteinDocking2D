from DeepProteinDocking2D.Models.BruteForce.utility_functions import *

def plot_pose(FFT_score, receptor, ligand, gt_rot, gt_txy, plot_count, stream_name):
    plt.close()
    plt.figure(figsize=(8, 8))
    # pred_rot, pred_txy = TorchDockingFFT().extract_transform(FFT_score)
    # print('extracted predicted indices', pred_rot, pred_txy)
    print('gt indices', gt_rot, gt_txy)
    # rmsd_out = RMSD(ligand, gt_rot, gt_txy, pred_rot, pred_txy).calc_rmsd()
    # print('RMSD', rmsd_out.item())

    pair = plot_assembly(receptor, ligand,
                         gt_rot,
                         gt_txy,
                         gt_rot, gt_txy)

    plt.imshow(pair.transpose())
    plt.title('Ground truth', loc='left')
    plt.title('Input')
    plt.title('Predicted pose', loc='right')
    # plt.text(225, 110, "RMSD = " + str(rmsd_out.item())[:5], backgroundcolor='w')
    plt.grid(False)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)
    # plt.savefig('figs/rmsd_and_poses/' + stream_name + '_docking_pose_example' + str(plot_count) + '_RMSD' + str(
    #     rmsd_out.item())[:4] + '.png')
    plt.show()


if __name__ == "__main__":

    dataset = 'docking_data_test_torchv1p10'
    data = read_pkl(dataset)
    # print(data)
    # print(data[0])
    # print(data[1])
    # print(data[0][0],data[0][1])
    for i in range(len(data)):
        receptor, ligand, gt_txy, gt_rot = data[i]
        plot_pose(None, receptor, ligand, gt_rot, gt_txy, plot_count=1, stream_name='dockingdatadebug')
