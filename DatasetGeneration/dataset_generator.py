import os
import sys
import numpy as np
import torch
import argparse
import _pickle as pkl

import matplotlib.pylab as plt
import seaborn as sea
sea.set_style("whitegrid")

from random import uniform
from complex_generator import generate_complex, rotate_ligand
from prot_generator import generate_protein
from docking import dock_global
from tqdm import tqdm


def generate_example(max_attempts=100):
    for i in range(max_attempts):
        receptor, ligand, t_correct, rotation_angle_correct = generate_complex()
        if t_correct is None:
            return None
        
        score, translation, rotation, all_scores = dock_global(receptor, ligand, num_angles=360, boundary_size=1)
        
        if np.linalg.norm(translation - t_correct)<5.0 and np.abs(rotation_angle_correct-rotation)<10.0:
            return receptor, ligand, t_correct, rotation_angle_correct

    return None

def generate_dataset(num_examples=1000, max_example_attempts=100):
    dataset = []
    for i in tqdm(range(num_examples)):
        example = generate_example(max_attempts=max_example_attempts)
        if example is None:
            continue
        dataset.append(example)
    return dataset

if __name__=='__main__':

    with open('toy_dataset_1000.pkl', 'wb') as fout:
        dataset = generate_dataset(num_examples=1000)
        pkl.dump(dataset, fout)

    
