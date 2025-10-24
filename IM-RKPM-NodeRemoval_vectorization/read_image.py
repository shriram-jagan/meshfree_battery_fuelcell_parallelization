import time

start_time = time.time()
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm

#########################
# read image in
#########################


def read_in_image(file_name):

    img_ = plt.imread(file_name)

    img_ = (
        np.flip(img_, axis=0)
    ).T  # it is an array(number of pixels along x * number of pixels along y)

    grain_id_counter = Counter(
        sum(img_.tolist(), [])
    )  # sum(img_.tolist(),[]) convert array with shape of 51*51 to 1d list with length 2601 ,return a library, grain id: number of pixel with this id

    unic_grain_id = []

    for key in grain_id_counter:
        unic_grain_id.append(key)

    # img_ is a 2d array, the ith row corresponding to the ith row from top of the image
    # the jth column correspondes to the jth row from left of the image. img_[i,j] is the
    # grayscale of ith row from top jth column from left, pixel ij.

    num_pixels_x = np.shape(img_)[0]
    num_pixels_y = np.shape(img_)[1]

    return img_, unic_grain_id, num_pixels_x, num_pixels_y
