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
import scipy
from scipy.spatial import ConvexHull

def get_random_points(num_points, xspan, yspan):
    points = [[uniform(*xspan), uniform(*yspan)]  for i in range(num_points)]
    points = np.array(points)
    return points

def is_p_inside_points_hull(hull, p):
    new_points = np.append(hull.points, p, axis=0)
    new_hull = ConvexHull(new_points)
    if list(hull.vertices) == list(new_hull.vertices):
        return True
    else:
        return False

def hull2array(hull, array, xspan, yspan):
    x_size = array.shape[0]
    y_size = array.shape[1]
    for i in range(x_size):
        for j in range(y_size):
            x = xspan[0] + i*float(xspan[1]-xspan[0])/float(x_size)
            y = yspan[0] + j*float(yspan[1]-yspan[0])/float(y_size)
            if is_p_inside_points_hull(hull, np.array([[x,y]])):
                array[i, j] = 1.0
            else:
                array[i, j] = 0.0
    return array

def generate_protein(size=50, debug=False):
    grid_coordinate_span = (0,10)
    points_coordinate_span = (1,9)
    num_points = 15

    points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
    hull = scipy.spatial.ConvexHull(points)
    prot = np.zeros((size, size))
    prot = hull2array(hull, prot, grid_coordinate_span, grid_coordinate_span)
    if debug:
        plt.matshow(prot.transpose(), extent=[*grid_coordinate_span,*grid_coordinate_span], origin='lower')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', color='white')
        
        plt.scatter(x=[points[i, 0] for i in range(points.shape[0])],
                    y=[points[i, 1] for i in range(points.shape[0])])
        plt.show()
    
    return prot

def test_hull_projection():
    prot = generate_protein(size=50, debug=True)
    plt.matshow(prot.transpose(), origin='lower')
    plt.show()

def test_protein_generation():
    N_prot = 5
    field = np.zeros((50,N_prot*50))
    for i in range(N_prot):
        prot = generate_protein(size=50, debug=False)
        field[:, i*50:(i+1)*50] = prot

    plt.matshow(field, origin='lower')
    plt.show()

if __name__=='__main__':
    test_hull_projection()
    test_protein_generation()
    