import scipy.ndimage as ndimage
import _pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def write_pkl(data, fileprefix):
    '''
    :param data:
    :param filename:
    '''
    print('\nwriting '+fileprefix+' to .pkl\n')
    with open(fileprefix+'.pkl', 'wb') as fout:
        pkl.dump(data, fout)
    fout.close()


def read_pkl(fileprefix):
    '''
    :param filename:
    :return: data
    '''
    print('\nreading '+fileprefix+'.pkl\n')
    with open(fileprefix+'.pkl', 'rb') as fin:
        data = pkl.load(fin)
    fin.close()
    return data


def write_txt(data, fileprefix):
    '''
    :param data:
    :param filename:
    :return: writes text file with
    '''
    print('\nwriting '+fileprefix+' to .txt\n')
    fout = open(fileprefix+'.txt', 'w')
    for example in data:
        fout.write(str(example)[1:-1] + '\n')
    fout.close()

def plot_coords(ax, poly, plot_alpha=0.25):
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=plot_alpha)


def plot_multipoly(multipolygon):
    plt.close()
    fig, ax = plt.subplots()
    ax.axis('equal')
    for poly in multipolygon:
        plot_coords(ax, poly)


def get_rot_mat(theta):
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]]).cuda()


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x


def rotate_gridligand(ligand, rotation_angle):
    ligand = ndimage.rotate(ligand, rotation_angle, reshape=False, order=3, mode='nearest', cval=0.0)
    return ligand


def translate_gridligand(ligand, tx, ty):
    ligand = ndimage.shift(ligand, [tx, ty], mode='wrap', order=3, cval=0.0)
    return ligand


def plot_assembly(receptor, ligand, rotation, translation, gt_rot, gt_txy):
    box_size = receptor.shape[-1]

    gt_rot = (gt_rot * 180.0/np.pi)
    rotation = (rotation * 180.0/np.pi)

    print('plotting pred values',rotation, translation)
    print('plotting gt values', gt_rot, gt_txy)

    receptor_copy = receptor * -100
    ligand_copy = ligand * 200
    padding = box_size//2
    if box_size < 100:
        receptor_copy = np.pad(receptor_copy, ((padding, padding), (padding, padding)), 'constant',
                          constant_values=((0, 0), (0, 0)))
        ligand_copy = np.pad(ligand_copy, ((padding, padding), (padding, padding)), 'constant',
                        constant_values=((0, 0), (0, 0)))

    gt_transformlig = rotate_gridligand(ligand_copy, gt_rot)
    gt_transformlig = translate_gridligand(gt_transformlig, gt_txy[0], gt_txy[1])
    gt_transformlig += receptor_copy

    transformligand = rotate_gridligand(ligand_copy, rotation)
    transformligand = translate_gridligand(transformligand, translation[0], translation[1])
    transformligand += receptor_copy

    inputshapes = receptor_copy + ligand_copy
    pair = np.vstack((gt_transformlig, inputshapes, transformligand))

    return pair


def plot_dataset(dataset, setname, dim0, dim1):
    fig, ax = plt.subplots(dim0, dim1, figsize=(15, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.8, hspace=0.8)
    count = 0
    plt.suptitle("Ground Truth Transform                Input Shapes                  Predicted Pose")
    for assembly in dataset:
        j = count % dim0
        k = count % dim1
        count += 1
        receptor, ligand, score, MAE, (rotation, translation), (truerot, truetxy), _ = assembly

        scoring = 'Score =' + str(score)[:5] + ' MAErot=' + str(MAE[0])[:5] + ' MAEtrans=' + str(MAE[1])[:5]

        pair = plot_assembly(receptor, ligand, rotation, translation, truerot, truetxy)

        ax[j,k].imshow(pair.transpose())
        ax[j,k].grid(None)
        ax[j,k].axis('Off')
        ax[j,k].text(5, 5, scoring, bbox={'facecolor': 'white', 'pad': 5}, fontsize=7)

    plt.tight_layout()
    plt.savefig('figs/'+setname + str(count)+'.png')
    plt.close()

if __name__ == '__main__':
    print('works')