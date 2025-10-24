import time

start_time = time.time()
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm

#########################
# read image in
#########################


def read_in_image(file_name, studied_physics, dimention):

    img_ = tifffile.imread(file_name)  # np array

    if studied_physics == "battery":
        img_ = (
            np.flip(img_, axis=0)
        ).T  # it is an array(number of pixels along x * number of pixels along y)

    grain_id_counter = Counter(img_.flatten())
    unic_grain_id = (
        []
    )  # save the unique grain IDs. For fuel cell, 0: pore, 1: electrolyte, 2: electrode.

    for key in grain_id_counter:
        unic_grain_id.append(int(key))

    num_pixels_xyz = []  # number of volxels in x y z directions.

    num_pixels_x = np.shape(img_)[0]
    num_pixels_y = np.shape(img_)[1]

    num_pixels_xyz.append(num_pixels_x)
    num_pixels_xyz.append(num_pixels_y)

    if dimention == 3:
        num_pixels_z = np.shape(img_)[2]
        num_pixels_xyz.append(num_pixels_z)

    return img_, unic_grain_id, num_pixels_xyz
