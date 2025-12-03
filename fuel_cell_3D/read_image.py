from collections import Counter

import tifffile
from common import np

#########################
# read image in
#########################


def read_in_image(file_name, studied_physics, dimension):

    img_ = tifffile.imread(file_name)  # np array

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

    if dimension == 3:
        num_pixels_z = np.shape(img_)[2]
        num_pixels_xyz.append(num_pixels_z)

    return img_, unic_grain_id, num_pixels_xyz
