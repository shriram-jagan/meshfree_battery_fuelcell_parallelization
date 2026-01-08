"""
Image reading module for fuel cell microstructure.

This module provides functionality to read voxelized microstructure data
from TIFF image files for solid oxide fuel cell simulations.
"""

from collections import Counter

import tifffile
from common import np


def read_in_image(file_name, studied_physics, dimension):
    """
    Read a voxelized microstructure image from a TIFF file.

    This function loads a 3D TIFF image containing phase labels for the
    fuel cell microstructure. Each voxel is labeled with a phase ID:

    - 0: Pore (gas phase)
    - 1: Electrolyte (ionic conductor)
    - 2: Electrode (electronic conductor)

    Parameters
    ----------
    file_name : str
        Path to the TIFF image file.
    studied_physics : str
        Type of physics being studied ("fuel cell" or "battery").
    dimension : int
        Problem dimension (2 or 3).

    Returns
    -------
    img_ : numpy.ndarray
        3D array of shape (nx, ny, nz) containing phase IDs at each voxel.
    unic_grain_id : list
        List of unique phase IDs found in the image.
    num_pixels_xyz : list
        List [nx, ny, nz] of voxel counts in each direction.

    Examples
    --------
    >>> img, phases, dims = read_in_image("M_3d_3phases_2K.tif", "fuel cell", 3)
    >>> print(f"Image shape: {img.shape}")
    >>> print(f"Phases found: {phases}")
    """
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
