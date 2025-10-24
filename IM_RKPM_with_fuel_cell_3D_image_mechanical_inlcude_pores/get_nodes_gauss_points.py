import time

start_time = time.time()
from collections import Counter

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import jit, njit
from numba.typed import List
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm


@jit(nopython=True)
def get_x_nodes_single_grain_battery(n_nodes, n_intervals, x_min, x_max, y_min, y_max):
    x_nodes = []
    for j in range(n_nodes):
        for i in range(n_nodes):
            x_nodes.append(
                [
                    x_min + (x_max - x_min) / n_intervals * i,
                    y_min + (y_max - y_min) / n_intervals * j,
                ]
            )
    return x_nodes


@jit
def get_x_nodes_fuel_cell_2d_toy(
    x_min_electrolyte,
    x_max_electrolyte,
    y_min_electrolyte,
    y_max_electrolyte,
    x_min_electrode,
    x_max_electrode,
    y_min_electrode,
    y_max_electrode,
    n_intervals,
):

    # whole domain:
    x_min = x_min_electrolyte
    x_max = x_max_electrode
    y_min = y_min_electrolyte
    y_max = y_max_electrolyte

    n_nodes = n_intervals + 1

    x_nodes_electrolyte = []
    x_nodes_electrode = []

    x_nodes_mechanical = (
        []
    )  # includes all nodes in electrolyte and electrode as well as nodes at interface. Interface nodes will be domain nodes for mechanicsl simulation

    point_source_coords = []  # node coordinates with source
    point_fixed_coords = []
    point_fixed_coords.append([x_min, y_min])

    nodes_id_left_electrolyte = []
    nodes_id_right_electrode = []

    cell_nodes_electrolyte_x = []
    cell_nodes_electrolyte_y = []
    cell_nodes_electrode_x = []
    cell_nodes_electrode_y = []

    cell_nodes_left_electrolyte_y = []
    cell_nodes_right_electrode_y = []

    nodes_id_electrolyte = 0
    nodes_id_electrode = 0

    # all nodes for mechanical
    for i in range(2 * n_nodes - 1):
        for j in range(n_nodes):
            x_nodes_mechanical.append(
                [
                    x_min + (x_max - x_min) / n_intervals / 2 * i,
                    y_min + (y_max - y_min) / n_intervals * j,
                ]
            )

    # in electrolyte or electrode
    for j in range(n_nodes):
        for i in range(n_nodes):
            x_nodes_electrolyte.append(
                [
                    x_min_electrolyte
                    + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                    y_min_electrolyte
                    + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                ]
            )
            x_nodes_electrode.append(
                [
                    x_min_electrode
                    + (x_max_electrode - x_min_electrode) / n_intervals * i,
                    y_min_electrode
                    + (y_max_electrode - y_min_electrode) / n_intervals * j,
                ]
            )

            if j < n_intervals and i < n_intervals:
                cell_nodes_electrolyte_x.append(
                    [
                        x_min_electrolyte
                        + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                        x_min_electrolyte
                        + (x_max_electrolyte - x_min_electrolyte)
                        / n_intervals
                        * (i + 1),
                        x_min_electrolyte
                        + (x_max_electrolyte - x_min_electrolyte)
                        / n_intervals
                        * (i + 1),
                        x_min_electrolyte
                        + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                    ]
                )
                cell_nodes_electrolyte_y.append(
                    [
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte)
                        / n_intervals
                        * (j + 1),
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte)
                        / n_intervals
                        * (j + 1),
                    ]
                )
                cell_nodes_electrode_x.append(
                    [
                        x_min_electrode
                        + (x_max_electrode - x_min_electrode) / n_intervals * i,
                        x_min_electrode
                        + (x_max_electrode - x_min_electrode) / n_intervals * (i + 1),
                        x_min_electrode
                        + (x_max_electrode - x_min_electrode) / n_intervals * (i + 1),
                        x_min_electrode
                        + (x_max_electrode - x_min_electrode) / n_intervals * i,
                    ]
                )
                cell_nodes_electrode_y.append(
                    [
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * j,
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * j,
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * (j + 1),
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * (j + 1),
                    ]
                )

            if i == 0 and j < n_intervals:
                cell_nodes_left_electrolyte_y.append(
                    [
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte)
                        / n_intervals
                        * (j + 1),
                    ]
                )
                nodes_id_left_electrolyte.append(nodes_id_electrolyte)

            if i == n_nodes - 1 and j < n_intervals:
                cell_nodes_right_electrode_y.append(
                    [
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * j,
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * (j + 1),
                    ]
                )
                nodes_id_right_electrode.append(nodes_id_electrode)
            if (j == 0 or j == n_nodes - 1) and (i == n_nodes - 1):
                point_source_coords.append(
                    [
                        x_min_electrolyte
                        + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                    ]
                )

            nodes_id_electrolyte += 1
            nodes_id_electrode += 1

    return (
        x_nodes_mechanical,
        x_nodes_electrolyte,
        x_nodes_electrode,
        point_source_coords,
        point_fixed_coords,
        nodes_id_left_electrolyte,
        nodes_id_right_electrode,
        cell_nodes_electrolyte_x,
        cell_nodes_electrolyte_y,
        cell_nodes_electrode_x,
        cell_nodes_electrode_y,
        cell_nodes_left_electrolyte_y,
        cell_nodes_right_electrode_y,
    )


# @jit
def get_x_nodes_fuel_cell_3d_toy(
    x_min_electrolyte,
    x_max_electrolyte,
    y_min_electrolyte,
    y_max_electrolyte,
    z_min_electrolyte,
    z_max_electrolyte,
    x_min_electrode,
    x_max_electrode,
    y_min_electrode,
    y_max_electrode,
    z_min_electrode,
    z_max_electrode,
    n_intervals,
):

    # whole domain:
    x_min = x_min_electrolyte
    x_max = x_max_electrolyte
    y_min = y_min_electrolyte
    y_max = y_max_electrode
    z_min = z_min_electrolyte
    z_max = z_max_electrolyte

    x_nodes_electrolyte = []
    x_nodes_electrode = []
    x_nodes_mechanical = []

    segments_source_coords = []
    segments_fixed_coords = []

    nodes_id_left_electrolyte = []  # with Diretchlet BC
    nodes_id_right_electrode = []

    nodes_id_electrolyte = 0
    nodes_id_electrode = 0

    cell_nodes_electrolyte_x = []
    cell_nodes_electrolyte_y = []
    cell_nodes_electrolyte_z = []

    cell_nodes_electrode_x = []
    cell_nodes_electrode_y = []
    cell_nodes_electrode_z = []

    cell_nodes_left_electrolyte_x = []
    cell_nodes_left_electrolyte_z = []
    cell_nodes_right_electrode_x = []
    cell_nodes_right_electrode_z = []

    n_nodes = n_intervals + 1

    # all nodes:
    for i in range(n_nodes):
        for j in range(2 * n_nodes - 1):
            for k in range(n_nodes):
                x_nodes_mechanical.append(
                    [
                        x_min + (x_max - x_min) / n_intervals * i,
                        y_min + (y_max - y_min) / 2 / n_intervals * j,
                        z_min + (z_max - z_min) / n_intervals * k,
                    ]
                )

                if j == 0 and k == 0 and i < n_intervals:
                    segments_fixed_coords.append(
                        [
                            x_min + (x_max - x_min) / n_intervals * i,
                            0,
                            0,
                            x_min + (x_max - x_min) / n_intervals * (i + 1),
                            0,
                            0,
                        ]
                    )

    # in electrolyte
    for j in range(n_nodes):
        for i in range(n_nodes):
            for k in range(n_nodes):
                x_nodes_electrolyte.append(
                    [
                        x_min_electrolyte
                        + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                        y_min_electrolyte
                        + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                        z_min_electrolyte
                        + (z_max_electrolyte - z_min_electrolyte) / n_intervals * k,
                    ]
                )
                x_nodes_electrode.append(
                    [
                        x_min_electrode
                        + (x_max_electrode - x_min_electrode) / n_intervals * i,
                        y_min_electrode
                        + (y_max_electrode - y_min_electrode) / n_intervals * j,
                        z_min_electrode
                        + (z_max_electrode - z_min_electrode) / n_intervals * k,
                    ]
                )

                if i < n_intervals and j < n_intervals and k < n_intervals:
                    cell_nodes_electrolyte_x.append(
                        [
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte)
                            / n_intervals
                            * (i + 1),
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte)
                            / n_intervals
                            * (i + 1),
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte)
                            / n_intervals
                            * (i + 1),
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte)
                            / n_intervals
                            * (i + 1),
                            x_min_electrolyte
                            + (x_max_electrolyte - x_min_electrolyte) / n_intervals * i,
                        ]
                    )

                    cell_nodes_electrolyte_y.append(
                        [
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte)
                            / n_intervals
                            * (j + 1),
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte)
                            / n_intervals
                            * (j + 1),
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte) / n_intervals * j,
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte)
                            / n_intervals
                            * (j + 1),
                            y_min_electrolyte
                            + (y_max_electrolyte - y_min_electrolyte)
                            / n_intervals
                            * (j + 1),
                        ]
                    )

                    cell_nodes_electrolyte_z.append(
                        [
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte) / n_intervals * k,
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte) / n_intervals * k,
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte) / n_intervals * k,
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte) / n_intervals * k,
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte)
                            / n_intervals
                            * (k + 1),
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte)
                            / n_intervals
                            * (k + 1),
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte)
                            / n_intervals
                            * (k + 1),
                            z_min_electrolyte
                            + (z_max_electrolyte - z_min_electrolyte)
                            / n_intervals
                            * (k + 1),
                        ]
                    )

                    cell_nodes_electrode_x.append(
                        [
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode) / n_intervals * i,
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode)
                            / n_intervals
                            * (i + 1),
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode)
                            / n_intervals
                            * (i + 1),
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode) / n_intervals * i,
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode) / n_intervals * i,
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode)
                            / n_intervals
                            * (i + 1),
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode)
                            / n_intervals
                            * (i + 1),
                            x_min_electrode
                            + (x_max_electrode - x_min_electrode) / n_intervals * i,
                        ]
                    )

                    cell_nodes_electrode_y.append(
                        [
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode) / n_intervals * j,
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode) / n_intervals * j,
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode)
                            / n_intervals
                            * (j + 1),
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode)
                            / n_intervals
                            * (j + 1),
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode) / n_intervals * j,
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode) / n_intervals * j,
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode)
                            / n_intervals
                            * (j + 1),
                            y_min_electrode
                            + (y_max_electrode - y_min_electrode)
                            / n_intervals
                            * (j + 1),
                        ]
                    )

                    cell_nodes_electrode_z.append(
                        [
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode) / n_intervals * k,
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode) / n_intervals * k,
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode) / n_intervals * k,
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode) / n_intervals * k,
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode)
                            / n_intervals
                            * (k + 1),
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode)
                            / n_intervals
                            * (k + 1),
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode)
                            / n_intervals
                            * (k + 1),
                            z_min_electrode
                            + (z_max_electrode - z_min_electrode)
                            / n_intervals
                            * (k + 1),
                        ]
                    )

                if j == 0:
                    nodes_id_left_electrolyte.append(nodes_id_electrolyte)
                    if i < n_intervals and k < n_intervals:
                        cell_nodes_left_electrolyte_x.append(
                            [
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * i,
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * (i + 1),
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * (i + 1),
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * i,
                            ]
                        )
                        cell_nodes_left_electrolyte_z.append(
                            [
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * k,
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * k,
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * (k + 1),
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * (k + 1),
                            ]
                        )

                if j == n_nodes - 1:
                    nodes_id_right_electrode.append(nodes_id_electrode)
                    if i < n_intervals and k < n_intervals:
                        cell_nodes_right_electrode_x.append(
                            [
                                x_min_electrode
                                + (x_max_electrode - x_min_electrode) / n_intervals * i,
                                x_min_electrode
                                + (x_max_electrode - x_min_electrode)
                                / n_intervals
                                * (i + 1),
                                x_min_electrode
                                + (x_max_electrode - x_min_electrode)
                                / n_intervals
                                * (i + 1),
                                x_min_electrode
                                + (x_max_electrode - x_min_electrode) / n_intervals * i,
                            ]
                        )
                        cell_nodes_right_electrode_z.append(
                            [
                                z_min_electrode
                                + (z_max_electrode - z_min_electrode) / n_intervals * k,
                                z_min_electrode
                                + (z_max_electrode - z_min_electrode) / n_intervals * k,
                                z_min_electrode
                                + (z_max_electrode - z_min_electrode)
                                / n_intervals
                                * (k + 1),
                                z_min_electrode
                                + (z_max_electrode - z_min_electrode)
                                / n_intervals
                                * (k + 1),
                            ]
                        )

                    if (i == 0 or i == n_nodes - 1) and k < n_intervals:
                        segments_source_coords.append(
                            [
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * i,
                                y_min_electrolyte
                                + (y_max_electrolyte - y_min_electrolyte)
                                / n_intervals
                                * j,
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * k,
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * i,
                                y_min_electrolyte
                                + (y_max_electrolyte - y_min_electrolyte)
                                / n_intervals
                                * j,
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * (k + 1),
                            ]
                        )
                    if (k == 0 or k == n_nodes - 1) and i < n_intervals:
                        segments_source_coords.append(
                            [
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * i,
                                y_min_electrolyte
                                + (y_max_electrolyte - y_min_electrolyte)
                                / n_intervals
                                * j,
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * k,
                                x_min_electrolyte
                                + (x_max_electrolyte - x_min_electrolyte)
                                / n_intervals
                                * (i + 1),
                                y_min_electrolyte
                                + (y_max_electrolyte - y_min_electrolyte)
                                / n_intervals
                                * j,
                                z_min_electrolyte
                                + (z_max_electrolyte - z_min_electrolyte)
                                / n_intervals
                                * k,
                            ]
                        )

                nodes_id_electrolyte += 1
                nodes_id_electrode += 1

    return (
        x_nodes_mechanical,
        x_nodes_electrolyte,
        x_nodes_electrode,
        segments_source_coords,
        segments_fixed_coords,
        nodes_id_left_electrolyte,
        nodes_id_right_electrode,
        cell_nodes_electrolyte_x,
        cell_nodes_electrolyte_y,
        cell_nodes_electrolyte_z,
        cell_nodes_electrode_x,
        cell_nodes_electrode_y,
        cell_nodes_electrode_z,
        cell_nodes_left_electrolyte_x,
        cell_nodes_left_electrolyte_z,
        cell_nodes_right_electrode_x,
        cell_nodes_right_electrode_z,
    )


def get_x_nodes_fuel_cell_3d_toy_image(
    x_min, x_max, y_min, y_max, z_min, z_max, num_pixels_xyz, img_
):
    # x_min... are the range of the whole domain.

    num_pixels_x = num_pixels_xyz[0]  # number of pixels/nodes along x
    num_pixels_y = num_pixels_xyz[1]
    num_pixels_z = num_pixels_xyz[2]

    # nodes in domain
    x_nodes_electrolyte = []
    x_nodes_electrode = []
    x_nodes_pore = []
    x_nodes_mechanical = []

    # nodes in each cell, used to calculate the gauss points in each cell
    cell_nodes_electrolyte_x = []
    cell_nodes_electrolyte_y = []
    cell_nodes_electrolyte_z = []
    cell_nodes_electrode_x = []
    cell_nodes_electrode_y = []
    cell_nodes_electrode_z = []
    cell_nodes_pore_x = []
    cell_nodes_pore_y = []
    cell_nodes_pore_z = []

    nodes_id_electrolyte = 0
    nodes_id_electrode = 0
    nodes_id_pore = 0

    # nodes in each cell on boundaries with Diretchlet BC
    cell_nodes_left_electrolyte_x = []  # with Diretchlet BC
    cell_nodes_left_electrolyte_z = []  # with Diretchlet BC
    cell_nodes_right_electrode_x = []
    cell_nodes_right_electrode_z = []
    cell_nodes_right_pore_x = []
    cell_nodes_right_pore_z = []

    nodes_id_left_electrolyte = []  # with Diretchlet BC
    nodes_id_right_electrode = []
    nodes_id_right_pore = []

    # segments on triple junctions or with flux, line integral of point source.
    segments_source = (
        []
    )  # n by 6 array, n is the number of segments with flux, 2 points on this segments, 6 coordinates for 3d
    segments_fixed = []  # n by 6 array, n is the number of segments on fixed edge.

    # at the interface of electrolyte/electrode and pore/electrode
    # cell_nodes_interface_electrode_electrolyte_electrolyte_x = []
    # cell_nodes_interface_electrode_electrolyte_electrolyte_y = []
    # cell_nodes_interface_electrode_electrolyte_electrolyte_z = []
    # cell_nodes_interface_electrode_pore_electrode_x = []
    # cell_nodes_interface_electrode_pore_electrode_y = []
    # cell_nodes_interface_electrode_pore_electrode_z = []

    # cell_nodes_interface_electrode_electrolyte_electrode_x = []
    # cell_nodes_interface_electrode_electrolyte_electrode_y = []
    # cell_nodes_interface_electrode_electrolyte_electrode_z = []
    # cell_nodes_interface_electrode_pore_pore_x = []
    # cell_nodes_interface_electrode_pore_pore_y = []
    # cell_nodes_interface_electrode_pore_pore_z = []

    cell_nodes_interface_electrode_electrolyte_x = []
    cell_nodes_interface_electrode_electrolyte_y = []
    cell_nodes_interface_electrode_electrolyte_z = []
    cell_nodes_interface_electrode_pore_x = []
    cell_nodes_interface_electrode_pore_y = []
    cell_nodes_interface_electrode_pore_z = []

    for i in range(num_pixels_x):
        for j in range(num_pixels_y):
            for k in range(num_pixels_z):
                if j == 0 and k == 0:
                    segments_fixed.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            0,
                            0,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            0,
                            0,
                        ]
                    )

                if img_[i, j, k] != 0:
                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_mechanical:
                        x_nodes_mechanical.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                # if in electrolyte domain
                if img_[i, j, k] == 2:

                    cell_nodes_electrolyte_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                        ]
                    )

                    cell_nodes_electrolyte_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                    cell_nodes_electrolyte_z.append(
                        [
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ]
                    )

                    if j == 0:
                        cell_nodes_left_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_left_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                    # check if each edge is a triple junction, edge 1
                    adjacent_pixel_index = np.array(
                        [[i, j - 1, k], [i, j - 1, k - 1], [i, j, k - 1]]
                    )  # 3 adjacent pixels

                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )

                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                ]
                            )

                    # edge 2
                    adjacent_pixel_index = np.array(
                        [[i + 1, j, k], [i + 1, j, k - 1], [i, j, k - 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                ]
                            )

                    # edge 3
                    adjacent_pixel_index = np.array(
                        [[i, j + 1, k], [i, j + 1, k - 1], [i, j, k - 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                ]
                            )

                    # edge 4
                    adjacent_pixel_index = np.array(
                        [[i - 1, j, k], [i - 1, j, k - 1], [i, j, k - 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                ]
                            )

                    # edge 5
                    adjacent_pixel_index = np.array(
                        [[i, j - 1, k], [i, j - 1, k + 1], [i, j, k + 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 6
                    adjacent_pixel_index = np.array(
                        [[i + 1, j, k], [i + 1, j, k + 1], [i, j, k + 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 7
                    adjacent_pixel_index = np.array(
                        [[i, j + 1, k], [i, j + 1, k + 1], [i, j, k + 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 8
                    adjacent_pixel_index = np.array(
                        [[i - 1, j, k], [i - 1, j, k + 1], [i, j, k + 1]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 9
                    adjacent_pixel_index = np.array(
                        [[i + 1, j - 1, k], [i, j - 1, k], [i + 1, j, k]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 10
                    adjacent_pixel_index = np.array(
                        [[i + 1, j, k], [i, j + 1, k], [i + 1, j + 1, k]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 11
                    adjacent_pixel_index = np.array(
                        [[i, j + 1, k], [i - 1, j, k], [i - 1, j + 1, k]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    # edge 12
                    adjacent_pixel_index = np.array(
                        [[i, j - 1, k], [i - 1, j, k], [i - 1, j - 1, k]]
                    )  # 3 adjacent pixels
                    filter_mask = (
                        np.all(
                            adjacent_pixel_index >= 0, axis=1
                        )  # all values non-negative
                        & (
                            adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                        )  # first column ≤ num_pixels_x
                        & (
                            adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                        )  # second column ≤ num_pixels_y
                        & (
                            adjacent_pixel_index[:, 2] <= num_pixels_z - 1
                        )  # third column ≤ num_pixels_z
                    )
                    filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                    unique_id_djacent_pixels = np.unique(
                        img_[tuple(filtered_adjacent_pixel_index.T)]
                    )
                    # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                    if (
                        0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                    ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                        if [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ] not in segments_source:
                            segments_source.append(
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * k,
                                    x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                                    z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                ]
                            )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        # if on left boundary
                        if j == 0:
                            nodes_id_left_electrolyte.append(nodes_id_electrolyte)

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        # if on left boundary
                        if j == 0:
                            nodes_id_left_electrolyte.append(nodes_id_electrolyte)

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # if on left boundary
                        if j == 0:
                            nodes_id_left_electrolyte.append(nodes_id_electrolyte)

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # if on left boundary
                        if j == 0:
                            nodes_id_left_electrolyte.append(nodes_id_electrolyte)

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                        nodes_id_electrolyte += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrolyte:
                        x_nodes_electrolyte.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                        nodes_id_electrolyte += 1

                    # get the interface, surface interface
                    # surface 1
                    surface_adjacent_pixel_index = np.array(
                        [i - 1, j, k]
                    )  # 3 adjacent pixels
                    if (i - 1) >= 0 and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 1:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_electrolyte_electrode_x.append([x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*i])
                        # cell_nodes_interface_electrode_electrolyte_electrode_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1)])

                    # surface 2
                    surface_adjacent_pixel_index = np.array(
                        [i + 1, j, k]
                    )  # 3 adjacent pixels
                    if (i + 1) < num_pixels_x and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 1:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_electrolyte_electrode_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 3
                    surface_adjacent_pixel_index = np.array(
                        [i, j - 1, k]
                    )  # 3 adjacent pixels
                    if (j - 1) >= 0 and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 1:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_electrolyte_electrode_x.append([x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*j])
                        # cell_nodes_interface_electrode_electrolyte_electrode_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 4
                    surface_adjacent_pixel_index = np.array(
                        [i, j + 1, k]
                    )  # 3 adjacent pixels
                    if (j + 1) < num_pixels_y and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 1:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_electrolyte_electrode_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_y.append([y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 5
                    surface_adjacent_pixel_index = np.array(
                        [i, j, k - 1]
                    )  # 3 adjacent pixels
                    if (k - 1) >= 0 and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 1:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k),
                            ]
                        )
                        # cell_nodes_interface_electrode_electrolyte_electrode_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*i])
                        # cell_nodes_interface_electrode_electrolyte_electrode_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 6
                    surface_adjacent_pixel_index = np.array(
                        [i, j, k + 1]
                    )  # 3 adjacent pixels
                    if (k + 1) < num_pixels_z and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 1:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_electrolyte_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            ]
                        )
                        cell_nodes_interface_electrode_electrolyte_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_electrolyte_electrode_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*(i+1)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_y.append([y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1)])
                        # cell_nodes_interface_electrode_electrolyte_electrode_z.append([z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1)])

                # if in electrode domain
                if img_[i, j, k] == 1:

                    cell_nodes_electrode_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                        ]
                    )

                    cell_nodes_electrode_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                    cell_nodes_electrode_z.append(
                        [
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ]
                    )

                    if j + 1 == num_pixels_y:
                        cell_nodes_right_electrode_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_right_electrode_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_electrode.append(nodes_id_electrode)
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_electrode.append(nodes_id_electrode)
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_electrode.append(nodes_id_electrode)
                        nodes_id_electrode += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_electrode:
                        x_nodes_electrode.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_electrode.append(nodes_id_electrode)
                        nodes_id_electrode += 1

                    # get the interface, surface interface
                    # surface 1
                    surface_adjacent_pixel_index = np.array(
                        [i - 1, j, k]
                    )  # 3 adjacent pixels
                    if (i - 1) >= 0 and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 0:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_interface_electrode_pore_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_pore_pore_x.append([x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*i])
                        # cell_nodes_interface_electrode_pore_pore_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j)])
                        # cell_nodes_interface_electrode_pore_pore_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1)])

                    # surface 2
                    surface_adjacent_pixel_index = np.array(
                        [i + 1, j, k]
                    )  # 3 adjacent pixels
                    if (i + 1) < num_pixels_x and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 0:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                            ]
                        )
                        cell_nodes_interface_electrode_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_pore_pore_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1)])
                        # cell_nodes_interface_electrode_pore_pore_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1)])
                        # cell_nodes_interface_electrode_pore_pore_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 3
                    surface_adjacent_pixel_index = np.array(
                        [i, j - 1, k]
                    )  # 3 adjacent pixels
                    if (j - 1) >= 0 and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 0:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_interface_electrode_pore_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                            ]
                        )
                        cell_nodes_interface_electrode_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_pore_pore_x.append([x_min+(x_max-x_min)/(num_pixels_x)*i, x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1)])
                        # cell_nodes_interface_electrode_pore_pore_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*j])
                        # cell_nodes_interface_electrode_pore_pore_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 4
                    surface_adjacent_pixel_index = np.array(
                        [i, j + 1, k]
                    )  # 3 adjacent pixels
                    if (j + 1) < num_pixels_x and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 0:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_pore_pore_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i)])
                        # cell_nodes_interface_electrode_pore_pore_y.append([y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1)])
                        # cell_nodes_interface_electrode_pore_pore_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 5
                    surface_adjacent_pixel_index = np.array(
                        [i, j, k - 1]
                    )  # 3 adjacent pixels
                    if (k - 1) >= 0 and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 0:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k),
                            ]
                        )
                        # cell_nodes_interface_electrode_pore_pore_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*i])
                        # cell_nodes_interface_electrode_pore_pore_y.append([y_min+(y_max-y_min)/(num_pixels_y)*j, y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j)])
                        # cell_nodes_interface_electrode_pore_pore_z.append([z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*k, z_min+(z_max-z_min)/(num_pixels_z)*(k), z_min+(z_max-z_min)/(num_pixels_z)*(k)])

                    # surface 6
                    surface_adjacent_pixel_index = np.array(
                        [i, j, k + 1]
                    )  # 3 adjacent pixels
                    if (k + 1) < num_pixels_x and img_[
                        tuple(surface_adjacent_pixel_index)
                    ] == 0:  # electrolyte/electrode interface
                        cell_nodes_interface_electrode_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_y.append(
                            [
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            ]
                        )
                        cell_nodes_interface_electrode_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # cell_nodes_interface_electrode_pore_pore_x.append([x_min+(x_max-x_min)/(num_pixels_x)*(i+1), x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*(i), x_min+(x_max-x_min)/(num_pixels_x)*(i+1)])
                        # cell_nodes_interface_electrode_pore_pore_y.append([y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j), y_min+(y_max-y_min)/(num_pixels_y)*(j+1), y_min+(y_max-y_min)/(num_pixels_y)*(j+1)])
                        # cell_nodes_interface_electrode_pore_pore_z.append([z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1), z_min+(z_max-z_min)/(num_pixels_z)*(k+1)])

                # if in pore domain
                if img_[i, j, k] == 0:

                    cell_nodes_pore_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                        ]
                    )

                    cell_nodes_pore_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                    cell_nodes_pore_z.append(
                        [
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * k,
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                        ]
                    )

                    if j + 1 == num_pixels_y:
                        cell_nodes_right_pore_x.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                            ]
                        )
                        cell_nodes_right_pore_z.append(
                            [
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_pore.append(nodes_id_pore)
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * k,
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * k,
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_pore.append(nodes_id_pore)
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_pore.append(nodes_id_pore)
                        nodes_id_pore += 1

                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                    ] not in x_nodes_pore:
                        x_nodes_pore.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                                z_min + (z_max - z_min) / (num_pixels_z) * (k + 1),
                            ]
                        )
                        # if on right boundary
                        if j + 1 == num_pixels_y:
                            nodes_id_right_pore.append(nodes_id_pore)
                        nodes_id_pore += 1
    return (
        x_nodes_mechanical,
        x_nodes_electrolyte,
        x_nodes_electrode,
        x_nodes_pore,
        segments_source,
        segments_fixed,
        nodes_id_left_electrolyte,
        nodes_id_right_electrode,
        nodes_id_right_pore,
        cell_nodes_electrolyte_x,
        cell_nodes_electrolyte_y,
        cell_nodes_electrolyte_z,
        cell_nodes_electrode_x,
        cell_nodes_electrode_y,
        cell_nodes_electrode_z,
        cell_nodes_pore_x,
        cell_nodes_pore_y,
        cell_nodes_pore_z,
        cell_nodes_left_electrolyte_x,
        cell_nodes_left_electrolyte_z,
        cell_nodes_right_electrode_x,
        cell_nodes_right_electrode_z,
        cell_nodes_right_pore_x,
        cell_nodes_right_pore_z,
        cell_nodes_interface_electrode_electrolyte_x,
        cell_nodes_interface_electrode_electrolyte_y,
        cell_nodes_interface_electrode_electrolyte_z,
        cell_nodes_interface_electrode_pore_x,
        cell_nodes_interface_electrode_pore_y,
        cell_nodes_interface_electrode_pore_z,
    )


def get_x_nodes_fuel_cell_2d_toy_image(
    x_min, x_max, y_min, y_max, num_pixels_xyz, img_
):
    # x_min... are the range of the whole domain.

    num_pixels_x = num_pixels_xyz[0]  # number of pixels/nodes along x
    num_pixels_y = num_pixels_xyz[1]

    # nodes in domain
    x_nodes_electrolyte = []
    x_nodes_electrode = []
    x_nodes_pore = []
    x_nodes_mechanical = []

    # nodes in each cell, used to calculate the gauss points in each cell
    cell_nodes_electrolyte_x = []
    cell_nodes_electrolyte_y = []
    cell_nodes_electrode_x = []
    cell_nodes_electrode_y = []
    cell_nodes_pore_x = []
    cell_nodes_pore_y = []

    nodes_id_electrolyte = 0
    nodes_id_electrode = 0
    nodes_id_pore = 0

    # nodes in each cell on boundaries with Diretchlet BC
    cell_nodes_left_electrolyte_y = []  # with Diretchlet BC
    cell_nodes_right_electrode_y = []
    cell_nodes_right_pore_y = []

    nodes_id_left_electrolyte = []  # with Diretchlet BC
    nodes_id_right_electrode = []
    nodes_id_right_pore = []

    # segments on triple junctions or with flux, line integral of point source.
    point_source = (
        []
    )  # n by 6 array, n is the number of segments with flux, 2 points on this segments, 6 coordinates for 3d

    point_fixed = []  # n by 6 array, n is the number of segments on fixed edge.
    point_fixed.append([x_min, y_min])  # bottom left corner is fixed.

    # at the interface of electrolyte/electrode and pore/electrode
    # cell_nodes_interface_electrode_electrolyte_electrolyte_x = []
    # cell_nodes_interface_electrode_electrolyte_electrolyte_y = []
    # cell_nodes_interface_electrode_electrolyte_electrolyte_z = []
    # cell_nodes_interface_electrode_pore_electrode_x = []
    # cell_nodes_interface_electrode_pore_electrode_y = []
    # cell_nodes_interface_electrode_pore_electrode_z = []

    # cell_nodes_interface_electrode_electrolyte_electrode_x = []
    # cell_nodes_interface_electrode_electrolyte_electrode_y = []
    # cell_nodes_interface_electrode_electrolyte_electrode_z = []
    # cell_nodes_interface_electrode_pore_pore_x = []
    # cell_nodes_interface_electrode_pore_pore_y = []
    # cell_nodes_interface_electrode_pore_pore_z = []

    cell_nodes_interface_electrode_electrolyte_x = []
    cell_nodes_interface_electrode_electrolyte_y = []
    cell_nodes_interface_electrode_pore_x = []
    cell_nodes_interface_electrode_pore_y = []

    for i in range(num_pixels_x):
        for j in range(num_pixels_y):

            if img_[i, j] != 0:
                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                ] not in x_nodes_mechanical:
                    x_nodes_mechanical.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                        ]
                    )

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j),
                ] not in x_nodes_mechanical:
                    x_nodes_mechanical.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        ]
                    )

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_mechanical:
                    x_nodes_mechanical.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_mechanical:
                    x_nodes_mechanical.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

            # if in electrolyte domain
            if img_[i, j] == 2:

                cell_nodes_electrolyte_x.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                    ]
                )

                cell_nodes_electrolyte_y.append(
                    [
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                    ]
                )

                if i == 0:
                    cell_nodes_left_electrolyte_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                # check if each edge is a triple junction, point 1
                adjacent_pixel_index = np.array(
                    [[i, j - 1], [i - 1, j], [i - 1, j - 1]]
                )  # 3 adjacent pixels

                filter_mask = (
                    np.all(adjacent_pixel_index >= 0, axis=1)  # all values non-negative
                    & (
                        adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                    )  # first column ≤ num_pixels_x
                    & (
                        adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                    )  # second column ≤ num_pixels_y
                )

                filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                unique_id_djacent_pixels = np.unique(
                    img_[tuple(filtered_adjacent_pixel_index.T)]
                )
                # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                if (
                    0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                    ] not in point_source:
                        point_source.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                            ]
                        )

                # point 2
                adjacent_pixel_index = np.array(
                    [[i + 1, j], [i + 1, j - 1], [i, j - 1]]
                )  # 3 adjacent pixels
                filter_mask = (
                    np.all(adjacent_pixel_index >= 0, axis=1)  # all values non-negative
                    & (
                        adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                    )  # first column ≤ num_pixels_x
                    & (
                        adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                    )  # second column ≤ num_pixels_y
                )
                filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                unique_id_djacent_pixels = np.unique(
                    img_[tuple(filtered_adjacent_pixel_index.T)]
                )
                # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                if (
                    0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                    ] not in point_source:
                        point_source.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * j,
                            ]
                        )

                # point 3
                adjacent_pixel_index = np.array(
                    [[i, j + 1], [i + 1, j + 1], [i + 1, j]]
                )  # 3 adjacent pixels
                filter_mask = (
                    np.all(adjacent_pixel_index >= 0, axis=1)  # all values non-negative
                    & (
                        adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                    )  # first column ≤ num_pixels_x
                    & (
                        adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                    )  # second column ≤ num_pixels_y
                )
                filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                unique_id_djacent_pixels = np.unique(
                    img_[tuple(filtered_adjacent_pixel_index.T)]
                )
                # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                if (
                    0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                    ] not in point_source:
                        point_source.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            ]
                        )

                # edge 4
                adjacent_pixel_index = np.array(
                    [[i, j + 1], [i - 1, j], [i - 1, j + 1]]
                )  # 3 adjacent pixels
                filter_mask = (
                    np.all(adjacent_pixel_index >= 0, axis=1)  # all values non-negative
                    & (
                        adjacent_pixel_index[:, 0] <= num_pixels_x - 1
                    )  # first column ≤ num_pixels_x
                    & (
                        adjacent_pixel_index[:, 1] <= num_pixels_y - 1
                    )  # second column ≤ num_pixels_y
                )
                filtered_adjacent_pixel_index = adjacent_pixel_index[filter_mask]
                unique_id_djacent_pixels = np.unique(
                    img_[tuple(filtered_adjacent_pixel_index.T)]
                )
                # check_if_edge = np.any(adjacent_pixel_index<0) or np.any(adjacent_pixel_index[:, 0] >= num_pixels_x) or np.any(adjacent_pixel_index[:, 1] >= num_pixels_y) or np.any(adjacent_pixel_index[:, 2] >= num_pixels_z)

                if (
                    0 in unique_id_djacent_pixels and 1 in unique_id_djacent_pixels
                ):  # or (1 in unique_id_djacent_pixels and check_if_edge):
                    if [
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                    ] not in point_source:
                        point_source.append(
                            [
                                x_min + (x_max - x_min) / (num_pixels_x) * i,
                                y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            ]
                        )

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                ] not in x_nodes_electrolyte:
                    x_nodes_electrolyte.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                        ]
                    )
                    # if on left boundary
                    if j == 0:
                        nodes_id_left_electrolyte.append(nodes_id_electrolyte)

                    nodes_id_electrolyte += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j),
                ] not in x_nodes_electrolyte:
                    x_nodes_electrolyte.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        ]
                    )
                    # if on left boundary
                    if j == 0:
                        nodes_id_left_electrolyte.append(nodes_id_electrolyte)

                    nodes_id_electrolyte += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_electrolyte:
                    x_nodes_electrolyte.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                    nodes_id_electrolyte += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_electrolyte:
                    x_nodes_electrolyte.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                    nodes_id_electrolyte += 1

                # get the interface, surface interface
                # surface 1
                surface_adjacent_pixel_index = np.array([i - 1, j])  # 3 adjacent pixels
                if (i - 1) >= 0 and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 1:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_electrolyte_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                        ]
                    )
                    cell_nodes_interface_electrode_electrolyte_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                # surface 2
                surface_adjacent_pixel_index = np.array([i + 1, j])  # 3 adjacent pixels
                if (i + 1) < num_pixels_x and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 1:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_electrolyte_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        ]
                    )
                    cell_nodes_interface_electrode_electrolyte_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                # surface 3
                surface_adjacent_pixel_index = np.array([i, j - 1])  # 3 adjacent pixels
                if (j - 1) >= 0 and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 1:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_electrolyte_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        ]
                    )
                    cell_nodes_interface_electrode_electrolyte_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        ]
                    )

                # surface 4
                surface_adjacent_pixel_index = np.array([i, j + 1])  # 3 adjacent pixels
                if (j + 1) < num_pixels_y and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 1:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_electrolyte_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        ]
                    )
                    cell_nodes_interface_electrode_electrolyte_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

            # if in electrode domain
            if img_[i, j] == 1:

                cell_nodes_electrode_x.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i),
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                    ]
                )

                cell_nodes_electrode_y.append(
                    [
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                    ]
                )

                if i + 1 == num_pixels_x:
                    cell_nodes_right_electrode_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                ] not in x_nodes_electrode:
                    x_nodes_electrode.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                        ]
                    )
                    nodes_id_electrode += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j),
                ] not in x_nodes_electrode:
                    x_nodes_electrode.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        ]
                    )
                    nodes_id_electrode += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_electrode:
                    x_nodes_electrode.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )
                    # if on right boundary
                    if j + 1 == num_pixels_y:
                        nodes_id_right_electrode.append(nodes_id_electrode)
                    nodes_id_electrode += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_electrode:
                    x_nodes_electrode.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )
                    # if on right boundary
                    if j + 1 == num_pixels_y:
                        nodes_id_right_electrode.append(nodes_id_electrode)
                    nodes_id_electrode += 1

                # get the interface, surface interface
                # surface 1
                surface_adjacent_pixel_index = np.array([i - 1, j])  # 3 adjacent pixels
                if (i - 1) >= 0 and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 0:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_pore_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                        ]
                    )
                    cell_nodes_interface_electrode_pore_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                # surface 2
                surface_adjacent_pixel_index = np.array([i + 1, j])  # 3 adjacent pixels
                if (i + 1) < num_pixels_x and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 0:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_pore_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        ]
                    )
                    cell_nodes_interface_electrode_pore_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                # surface 3
                surface_adjacent_pixel_index = np.array([i, j - 1])  # 3 adjacent pixels
                if (j - 1) >= 0 and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 0:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_pore_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        ]
                    )
                    cell_nodes_interface_electrode_pore_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        ]
                    )

                # surface 4
                surface_adjacent_pixel_index = np.array([i, j + 1])  # 3 adjacent pixels
                if (j + 1) < num_pixels_x and img_[
                    tuple(surface_adjacent_pixel_index)
                ] == 0:  # electrolyte/electrode interface
                    cell_nodes_interface_electrode_pore_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        ]
                    )
                    cell_nodes_interface_electrode_pore_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

            # if in pore domain
            if img_[i, j] == 0:

                cell_nodes_pore_x.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x) * (i),
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                        x_min + (x_max - x_min) / (num_pixels_x) * i,
                    ]
                )

                cell_nodes_pore_y.append(
                    [
                        y_min + (y_max - y_min) / (num_pixels_y) * j,
                        y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                    ]
                )

                if i + 1 == num_pixels_x:
                    cell_nodes_right_pore_y.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * j,
                ] not in x_nodes_pore:
                    x_nodes_pore.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * j,
                        ]
                    )
                    nodes_id_pore += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j),
                ] not in x_nodes_pore:
                    x_nodes_pore.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j),
                        ]
                    )
                    nodes_id_pore += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_pore:
                    x_nodes_pore.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )
                    # if on right boundary
                    if j + 1 == num_pixels_y:
                        nodes_id_right_pore.append(nodes_id_pore)
                    nodes_id_pore += 1

                if [
                    x_min + (x_max - x_min) / (num_pixels_x) * i,
                    y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                ] not in x_nodes_pore:
                    x_nodes_pore.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            y_min + (y_max - y_min) / (num_pixels_y) * (j + 1),
                        ]
                    )
                    # if on right boundary
                    if j + 1 == num_pixels_y:
                        nodes_id_right_pore.append(nodes_id_pore)
                    nodes_id_pore += 1

    return (
        x_nodes_mechanical,
        x_nodes_electrolyte,
        x_nodes_electrode,
        x_nodes_pore,
        point_source,
        point_fixed,
        nodes_id_left_electrolyte,
        nodes_id_right_electrode,
        nodes_id_right_pore,
        cell_nodes_electrolyte_x,
        cell_nodes_electrolyte_y,
        cell_nodes_electrode_x,
        cell_nodes_electrode_y,
        cell_nodes_pore_x,
        cell_nodes_pore_y,
        cell_nodes_left_electrolyte_y,
        cell_nodes_right_electrode_y,
        cell_nodes_right_pore_y,
        cell_nodes_interface_electrode_electrolyte_x,
        cell_nodes_interface_electrode_electrolyte_y,
        cell_nodes_interface_electrode_pore_x,
        cell_nodes_interface_electrode_pore_y,
    )


@jit
def get_x_nodes_multi_grain(x_min, x_max, y_min, y_max, num_pixels_xyz, img_):
    num_pixels_x = num_pixels_xyz[0]
    num_pixels_y = num_pixels_xyz[1]
    # define initial RPK nodes
    x_nodes_ini = []
    for j in range(num_pixels_x):
        for i in range(num_pixels_y):
            x_nodes_ini.append(
                [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * j,
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * i,
                ]
            )

    # go through all cells, partition cells if needed
    num_rec_cell = 0
    num_tri_cell = 0

    cell_nodes_list = []  # cell_nodes_list[i] is all nodes coordinates of cell i,
    grain_id = []  # grain id of each cell,
    cell_shape = []  # shape of each cell, triangle ('tri') or rectangle ('rec'),

    bottom_boundary_cell_nodes_list = (
        []
    )  # corresponding to bottom, right, top, left boundaries,
    right_boundary_cell_nodes_list = []
    top_boundary_cell_nodes_list = []
    left_boundary_cell_nodes_list = []

    grain_id_left = []
    grain_id_right = []
    grain_id_top = []
    grain_id_bottom = []

    x_nodes_added = []

    x_nodes = []

    nodes_grain_id = []

    repeated_vertex = (
        []
    )  # when do the gauss integral, the triangle element is treated as rectangle. One of the vertex of triangle was repeated (the first , or the third verex)

    interface_segments = []  # all interface segments

    # go through all nodes, number of nodes on x and y directions are equal to the number of pixels along x y.

    for j in range(num_pixels_y - 1):
        for i in range(num_pixels_x - 1):

            if [
                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
            ] not in x_nodes:
                x_nodes.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                        y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                    ]
                )
                nodes_grain_id.append(img_[i, j])
            if [
                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
            ] not in x_nodes:
                x_nodes.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                    ]
                )
                nodes_grain_id.append(img_[i + 1, j])
            if [
                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
            ] not in x_nodes:
                x_nodes.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                        y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                    ]
                )
                nodes_grain_id.append(img_[i + 1, j + 1])
            if [
                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
            ] not in x_nodes:
                x_nodes.append(
                    [
                        x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                        y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                    ]
                )
                nodes_grain_id.append(img_[i, j + 1])

            added_nodes_number = 0  # number of added nodes for each cell

            add_node_bottom = "False"
            add_node_right = "False"
            add_node_top = "False"
            add_node_left = "False"

            if img_[i, j] != img_[i + 1, j]:
                added_nodes_number = added_nodes_number + 1
                add_node_bottom = "True"
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                ] not in x_nodes_added:
                    x_nodes_added.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                        ]
                    )
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                ] not in x_nodes:
                    x_nodes.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                        ]
                    )
                    nodes_grain_id.append(
                        img_[i + 1, j]
                    )  # the gain id of nodes on interface does not matter
            if j == 0:
                if add_node_bottom == "True":
                    bottom_boundary_cell_nodes_list.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                        ]
                    )
                    bottom_boundary_cell_nodes_list.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                        ]
                    )
                    grain_id_bottom.append(img_[i, j])
                    grain_id_bottom.append(img_[i + 1, j])
                else:
                    bottom_boundary_cell_nodes_list.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                        ]
                    )
                    grain_id_bottom.append(img_[i, j])

            if img_[i + 1, j] != img_[i + 1, j + 1]:
                added_nodes_number = added_nodes_number + 1
                add_node_right = "True"
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                ] not in x_nodes_added:
                    x_nodes_added.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ]
                    )
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                ] not in x_nodes:
                    x_nodes.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ]
                    )
                    nodes_grain_id.append(img_[i + 1, j])
            if i == num_pixels_x - 2:
                if add_node_right == "True":
                    right_boundary_cell_nodes_list.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ]
                    )
                    right_boundary_cell_nodes_list.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ]
                    )
                    grain_id_right.append(img_[i + 1, j])
                    grain_id_right.append(img_[i + 1, j + 1])
                else:
                    right_boundary_cell_nodes_list.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ]
                    )
                    grain_id_right.append(img_[i + 1, j])

            if img_[i + 1, j + 1] != img_[i, j + 1]:
                added_nodes_number = added_nodes_number + 1
                add_node_top = "True"
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                ] not in x_nodes_added:
                    x_nodes_added.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ]
                    )
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                ] not in x_nodes:
                    x_nodes.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ]
                    )
                    nodes_grain_id.append(img_[i + 1, j])

            if j == num_pixels_y - 2:
                if add_node_top == "True":
                    top_boundary_cell_nodes_list.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                        ]
                    )
                    top_boundary_cell_nodes_list.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                        ]
                    )
                    grain_id_top.append(img_[i, j + 1])
                    grain_id_top.append(img_[i + 1, j + 1])
                else:
                    top_boundary_cell_nodes_list.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                        ]
                    )
                    grain_id_top.append(img_[i + 1, j + 1])

            if img_[i, j] != img_[i, j + 1]:
                added_nodes_number = added_nodes_number + 1
                add_node_left = "True"
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                ] not in x_nodes_added:
                    x_nodes_added.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ]
                    )
                if [
                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                    y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                ] not in x_nodes:
                    x_nodes.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ]
                    )
                    nodes_grain_id.append(img_[i + 1, j])

            if i == 0:
                if add_node_left == "True":
                    left_boundary_cell_nodes_list.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ]
                    )
                    left_boundary_cell_nodes_list.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ]
                    )
                    grain_id_left.append(img_[i, j])
                    grain_id_left.append(img_[i, j + 1])
                else:
                    left_boundary_cell_nodes_list.append(
                        [
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ]
                    )
                    grain_id_left.append(img_[i, j])

            # if no node should be added
            if added_nodes_number == 0:  # or added_nodes_number==1:
                cell_nodes_list.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                    ]
                )

                grain_id.append(img_[i, j])

                cell_shape.append("rec")
                repeated_vertex.append("No")
                num_rec_cell = num_rec_cell + 1

            if added_nodes_number == 2:  # interface of two different grains

                # cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                # grain_id.append(img_[i,j])
                # cell_shape.append('rec')
                # num_rec_cell  = num_rec_cell + 1

                if (add_node_bottom == "True" and add_node_top == "True") or (
                    add_node_left == "True" and add_node_right == "True"
                ):
                    # split into four squares
                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    if add_node_bottom == "True" and add_node_top == "True":
                        interface_segments.append(
                            [
                                [
                                    x_min
                                    + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                    y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                                ],
                                [
                                    x_min
                                    + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                    y_min
                                    + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                                ],
                            ]
                        )

                    if add_node_left == "True" and add_node_right == "True":
                        interface_segments.append(
                            [
                                [
                                    x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                    y_min
                                    + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                                ],
                                [
                                    x_min
                                    + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                    y_min
                                    + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                                ],
                            ]
                        )

                # split into 3 rectangle cells two triangle cells
                if add_node_bottom == "True" and add_node_right == "True":
                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * i,
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("tri")

                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )

                if add_node_bottom == "True" and add_node_left == "True":

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    # if i == 0 and j == num_pixels_y-2:
                    #     print(([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                    #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                    #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]]))
                    #     print(len(cell_nodes_list))
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )

                if add_node_top == "True" and add_node_left == "True":

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )

                if add_node_top == "True" and add_node_right == "True":

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )

            if added_nodes_number == 3:  # interface of 3 different grains

                # cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                #                             [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                #                             [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                #                             [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                # grain_id.append(img_[i,j])
                # cell_shape.append('rec')
                # num_rec_cell  = num_rec_cell + 1

                if add_node_left == "False":
                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )

                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )

                if add_node_bottom == "False":
                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )

                if add_node_right == "False":
                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )

                if add_node_top == "False":
                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("rec")
                    repeated_vertex.append("No")
                    num_rec_cell = num_rec_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j + 1])
                    cell_shape.append("tri")
                    repeated_vertex.append("first")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    cell_nodes_list.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )
                    grain_id.append(img_[i + 1, j])
                    cell_shape.append("tri")
                    repeated_vertex.append("three")
                    num_tri_cell = num_tri_cell + 1

                    interface_segments.append(
                        [
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                            ],
                        ]
                    )
                    interface_segments.append(
                        [
                            [
                                x_min
                                + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                                y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                            ],
                            [
                                x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                                y_min
                                + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                            ],
                        ]
                    )

            if added_nodes_number == 4:  # interface of 3 or 4 different grains

                # cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                #                                 [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                #                                 [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                #                                 [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                # grain_id.append(img_[i,j])
                # cell_shape.append('rec')
                # num_rec_cell  = num_rec_cell + 1

                cell_nodes_list.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                    ]
                )
                grain_id.append(img_[i, j])
                cell_shape.append("rec")
                repeated_vertex.append("No")
                num_rec_cell = num_rec_cell + 1

                cell_nodes_list.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                    ]
                )
                grain_id.append(img_[i + 1, j])
                cell_shape.append("rec")
                repeated_vertex.append("No")
                num_rec_cell = num_rec_cell + 1

                cell_nodes_list.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                    ]
                )
                grain_id.append(img_[i + 1, j + 1])
                cell_shape.append("rec")
                repeated_vertex.append("No")
                num_rec_cell = num_rec_cell + 1

                cell_nodes_list.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                    ]
                )
                grain_id.append(img_[i, j + 1])
                cell_shape.append("rec")
                repeated_vertex.append("No")
                num_rec_cell = num_rec_cell + 1

                interface_segments.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * j,
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 0.5),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 1),
                        ],
                    ]
                )
                # interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                #                         [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                interface_segments.append(
                    [
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                        [
                            x_min + (x_max - x_min) / (num_pixels_x - 1) * (i + 1),
                            y_min + (y_max - y_min) / (num_pixels_y - 1) * (j + 0.5),
                        ],
                    ]
                )
                # interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                #                         [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])

    x_nodes = np.array(x_nodes)
    nodes_grain_id = np.array(nodes_grain_id)

    # interface_segments

    return (
        cell_nodes_list,
        grain_id,
        grain_id_bottom,
        grain_id_top,
        grain_id_left,
        grain_id_right,
        cell_shape,
        num_rec_cell,
        num_tri_cell,
        x_nodes,
        nodes_grain_id,
        bottom_boundary_cell_nodes_list,
        right_boundary_cell_nodes_list,
        top_boundary_cell_nodes_list,
        left_boundary_cell_nodes_list,
        repeated_vertex,
        interface_segments,
    )


# get all gauss points in domain, 3d domain
def x_G_and_def_J_time_weight_structured_3d(
    n_intervals, x_min, x_max, y_min, y_max, z_min, z_max, x_G_domain, weight_G_domain
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian
    for n in range(n_intervals):
        for m in range(n_intervals):
            for nm in range(n_intervals):
                # in the mnnm (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
                x_ver_mn = np.array(
                    [
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + (m) * (x_max - x_min) / n_intervals,
                        x_min + m * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + (m) * (x_max - x_min) / n_intervals,
                        x_min + m * (x_max - x_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                y_ver_mn = np.array(
                    [
                        y_min + n * (y_max - y_min) / n_intervals,
                        y_min + (n + 1) * (y_max - y_min) / n_intervals,
                        y_min + (n + 1) * (y_max - y_min) / n_intervals,
                        y_min + (n) * (y_max - y_min) / n_intervals,
                        y_min + n * (y_max - y_min) / n_intervals,
                        y_min + (n + 1) * (y_max - y_min) / n_intervals,
                        y_min + (n + 1) * (y_max - y_min) / n_intervals,
                        y_min + (n) * (y_max - y_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                z_ver_mn = np.array(
                    [
                        z_min + nm * (z_max - z_min) / n_intervals,
                        z_min + nm * (z_max - z_min) / n_intervals,
                        z_min + nm * (z_max - z_min) / n_intervals,
                        z_min + nm * (z_max - z_min) / n_intervals,
                        z_min + (nm + 1) * (z_max - z_min) / n_intervals,
                        z_min + (nm + 1) * (z_max - z_min) / n_intervals,
                        z_min + (nm + 1) * (z_max - z_min) / n_intervals,
                        z_min + (nm + 1) * (z_max - z_min) / n_intervals,
                    ]
                )
                # calculate the cy coordinates of gauss points in current integration domain
                for k in range(len(x_G_domain)):

                    x_G_mn_k = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    (1 + x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    y_G_mn_k = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    (1 + x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    z_G_mn_k = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    (1 + x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 + x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0])
                                    * (1 - x_G_domain[k][1])
                                    * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )

                    x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

                    J11 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J12 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J13 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J21 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    J22 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    J23 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    J31 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )
                    J32 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                                    -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                    -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )
                    J33 = (
                        1.0
                        / 8.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )

                    det_J_time_weight.append(
                        np.linalg.det(
                            np.array(
                                [[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]]
                            )
                        )
                        * weight_G_domain[k]
                    )

    return x_G, det_J_time_weight


# get all gauss points in domain, 2d domain
def x_G_b_and_det_J_b_time_weight_structured_3d(
    n_intervals,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    face_id,
    x_G_domain,
    weight_G_domain,
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian

    if face_id == 0 or face_id == 2:

        for n in range(n_intervals):
            for m in range(n_intervals):
                # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
                x_ver_mn = np.array(
                    [
                        x_min + m * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + m * (x_max - x_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                z_ver_mn = np.array(
                    [
                        z_min + n * (z_max - z_min) / n_intervals,
                        z_min + n * (z_max - z_min) / n_intervals,
                        z_min + (n + 1) * (z_max - z_min) / n_intervals,
                        z_min + (n + 1) * (z_max - z_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                # calculate the cy coordinates of gauss points in current integration domain
                for k in range(len(x_G_domain)):

                    x_G_mn_k = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    z_G_mn_k = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )
                    if face_id == 0:
                        y_G_mn_k = y_min
                    else:
                        y_G_mn_k = y_max
                    x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])
                    J1 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][1]),
                                    (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][1]),
                                    (-1 - x_G_domain[k][1]),
                                ]
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J2 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][1]),
                                    (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][1]),
                                    (-1 - x_G_domain[k][1]),
                                ]
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )
                    J3 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][0]),
                                    (-1 - x_G_domain[k][0]),
                                    (1 + x_G_domain[k][0]),
                                    (1 - x_G_domain[k][0]),
                                ]
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J4 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][0]),
                                    (-1 - x_G_domain[k][0]),
                                    (1 + x_G_domain[k][0]),
                                    (1 - x_G_domain[k][0]),
                                ]
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )

                    det_J_time_weight.append(
                        np.linalg.det(np.array([[J1, J2], [J3, J4]]))
                        * weight_G_domain[k]
                    )
    if face_id == 1 or face_id == 3:

        for n in range(n_intervals):
            for m in range(n_intervals):
                # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
                y_ver_mn = np.array(
                    [
                        y_min + m * (y_max - y_min) / n_intervals,
                        y_min + (m + 1) * (y_max - y_min) / n_intervals,
                        y_min + (m + 1) * (y_max - y_min) / n_intervals,
                        y_min + m * (y_max - y_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                z_ver_mn = np.array(
                    [
                        z_min + n * (z_max - z_min) / n_intervals,
                        z_min + n * (z_max - z_min) / n_intervals,
                        z_min + (n + 1) * (z_max - z_min) / n_intervals,
                        z_min + (n + 1) * (z_max - z_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                # calculate the cy coordinates of gauss points in current integration domain
                for k in range(len(x_G_domain)):

                    y_G_mn_k = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    z_G_mn_k = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )
                    if face_id == 3:
                        x_G_mn_k = x_min
                    else:
                        x_G_mn_k = x_max
                    x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])
                    J1 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][1]),
                                    (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][1]),
                                    (-1 - x_G_domain[k][1]),
                                ]
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    J2 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][1]),
                                    (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][1]),
                                    (-1 - x_G_domain[k][1]),
                                ]
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )
                    J3 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][0]),
                                    (-1 - x_G_domain[k][0]),
                                    (1 + x_G_domain[k][0]),
                                    (1 - x_G_domain[k][0]),
                                ]
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    J4 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][0]),
                                    (-1 - x_G_domain[k][0]),
                                    (1 + x_G_domain[k][0]),
                                    (1 - x_G_domain[k][0]),
                                ]
                            ),
                            np.transpose(z_ver_mn),
                        )
                    )

                    det_J_time_weight.append(
                        np.linalg.det(np.array([[J1, J2], [J3, J4]]))
                        * weight_G_domain[k]
                    )
    if face_id == 4 or face_id == 5:

        for n in range(n_intervals):
            for m in range(n_intervals):
                # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
                x_ver_mn = np.array(
                    [
                        x_min + m * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + (m + 1) * (x_max - x_min) / n_intervals,
                        x_min + m * (x_max - x_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                y_ver_mn = np.array(
                    [
                        y_min + n * (y_max - y_min) / n_intervals,
                        y_min + n * (y_max - y_min) / n_intervals,
                        y_min + (n + 1) * (y_max - y_min) / n_intervals,
                        y_min + (n + 1) * (y_max - y_min) / n_intervals,
                    ],
                    dtype=np.float64,
                )
                # calculate the cy coordinates of gauss points in current integration domain
                for k in range(len(x_G_domain)):

                    x_G_mn_k = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    y_G_mn_k = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                    (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                ],
                                dtype=np.float64,
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    if face_id == 4:
                        z_G_mn_k = z_min
                    else:
                        z_G_mn_k = z_max
                    x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])
                    J1 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][1]),
                                    (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][1]),
                                    (-1 - x_G_domain[k][1]),
                                ]
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J2 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][1]),
                                    (1 - x_G_domain[k][1]),
                                    (1 + x_G_domain[k][1]),
                                    (-1 - x_G_domain[k][1]),
                                ]
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )
                    J3 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][0]),
                                    (-1 - x_G_domain[k][0]),
                                    (1 + x_G_domain[k][0]),
                                    (1 - x_G_domain[k][0]),
                                ]
                            ),
                            np.transpose(x_ver_mn),
                        )
                    )
                    J4 = (
                        1.0
                        / 4.0
                        * np.dot(
                            np.array(
                                [
                                    -(1 - x_G_domain[k][0]),
                                    (-1 - x_G_domain[k][0]),
                                    (1 + x_G_domain[k][0]),
                                    (1 - x_G_domain[k][0]),
                                ]
                            ),
                            np.transpose(y_ver_mn),
                        )
                    )

                    det_J_time_weight.append(
                        np.linalg.det(np.array([[J1, J2], [J3, J4]]))
                        * weight_G_domain[k]
                    )

    return x_G, det_J_time_weight


@jit
def x_G_and_det_J_structured_line_3d(
    n_boundaries,
    n_intervals,
    x_interface,
    y_max,
    y_min,
    z_max,
    z_min,
    x_G_line,
    weight_G_line,
):

    x_G_b_line = []
    det_J_b_time_weight_line = []  # determin of jacobian

    for i in range(n_boundaries):
        for j in range(n_intervals):
            if i == 0:  # bottom boundary
                y_ver_b = np.array(
                    [
                        y_min + (y_max - y_min) / n_intervals * j,
                        y_min + (y_max - y_min) / n_intervals * (j + 1),
                    ]
                )

                for k in range(len(x_G_line)):
                    y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_line[k] + (
                        y_ver_b[1] + y_ver_b[0]
                    ) / 2
                    z_G_ij_k = z_min
                    x_G_ij_k = x_interface
                    x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                    det_J_b_time_weight_line.append(
                        (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_line[k]
                    )

            if i == 1:  # right boundary
                z_ver_b = np.array(
                    [
                        z_min + (z_max - z_min) / n_intervals * j,
                        z_min + (z_max - z_min) / n_intervals * (j + 1),
                    ]
                )

                for k in range(len(x_G_line)):
                    x_G_ij_k = x_interface
                    y_G_ij_k = y_max
                    z_G_ij_k = (z_ver_b[1] - z_ver_b[0]) / 2 * x_G_line[k] + (
                        z_ver_b[1] + z_ver_b[0]
                    ) / 2
                    x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                    det_J_b_time_weight_line.append(
                        (z_ver_b[1] - z_ver_b[0]) / 2 * weight_G_line[k]
                    )

                """
                since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
                for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jacobian term
                if we integral from left to right, ds = dx
                """
            if i == 2:  # top boundary
                y_ver_b = np.array(
                    [
                        y_min + (y_max - y_min) / n_intervals * j,
                        y_min + (y_max - y_min) / n_intervals * (j + 1),
                    ]
                )
                # if x_ver_b = np.array([x_max-(x_max-x_min)/n_intervals*j, x_max-(x_max-x_min)/n_intervals*(j+1)]), we integral from right to left, det_J_b_time_weight should be -((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])

                for k in range(len(x_G_line)):
                    y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_line[k] + (
                        y_ver_b[1] + y_ver_b[0]
                    ) / 2  # if
                    x_G_ij_k = x_interface
                    z_G_ij_k = z_max
                    x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                    det_J_b_time_weight_line.append(
                        (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_line[k]
                    )

                """
                since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
                for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
                """
            if i == 3:  # left boundary
                z_ver_b = np.array(
                    [
                        z_min + (z_max - z_min) / n_intervals * j,
                        z_min + (z_max - z_min) / n_intervals * (j + 1),
                    ]
                )
                # if y_ver_b = np.array([y_max-(y_max-y_min)/n_intervals*j, y_max-(y_max-y_min)/n_intervals*(j+1)]), we integral from top to bottom, det_J_b_time_weight should be -((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

                for k in range(len(x_G_line)):
                    x_G_ij_k = x_interface
                    y_G_ij_k = y_min
                    z_G_ij_k = (z_ver_b[1] - z_ver_b[0]) / 2 * x_G_line[k] + (
                        z_ver_b[1] + z_ver_b[0]
                    ) / 2
                    x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                    det_J_b_time_weight_line.append(
                        (z_ver_b[1] - z_ver_b[0]) / 2 * weight_G_line[k]
                    )
    return x_G_b_line, det_J_b_time_weight_line


# get all gauss points in domain, 2d domain
def x_G_and_def_J_time_weight_structured(
    n_intervals, x_min, x_max, y_min, y_max, x_G_domain, weight_G_domain
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian
    for n in range(n_intervals):
        for m in range(n_intervals):
            # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
            x_ver_mn = np.array(
                [
                    x_min + m * (x_max - x_min) / n_intervals,
                    x_min + (m + 1) * (x_max - x_min) / n_intervals,
                    x_min + (m + 1) * (x_max - x_min) / n_intervals,
                    x_min + m * (x_max - x_min) / n_intervals,
                ],
                dtype=np.float64,
            )
            y_ver_mn = np.array(
                [
                    y_min + n * (y_max - y_min) / n_intervals,
                    y_min + n * (y_max - y_min) / n_intervals,
                    y_min + (n + 1) * (y_max - y_min) / n_intervals,
                    y_min + (n + 1) * (y_max - y_min) / n_intervals,
                ],
                dtype=np.float64,
            )
            # calculate the cy coordinates of gauss points in current integration domain
            for k in range(len(x_G_domain)):

                x_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                y_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                x_G.append([x_G_mn_k, y_G_mn_k])
                J1 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J2 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                J3 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J4 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )

                det_J_time_weight.append(
                    np.linalg.det(np.array([[J1, J2], [J3, J4]])) * weight_G_domain[k]
                )

    return x_G, det_J_time_weight


# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian on boundaries


@jit
def x_G_b_and_det_J_b_structured(
    n_boundaries,
    n_intervals,
    x_min,
    x_max,
    y_min,
    y_max,
    x_G_boundary,
    weight_G_boundary,
):

    x_G_b = []
    det_J_b_time_weight = []  # determin of jacobian

    for i in range(n_boundaries):
        for j in range(n_intervals):
            if i == 0:  # bottom boundary
                x_ver_b = np.array(
                    [
                        x_min + (x_max - x_min) / n_intervals * j,
                        x_min + (x_max - x_min) / n_intervals * (j + 1),
                    ]
                )

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = (x_ver_b[1] - x_ver_b[0]) / 2 * x_G_boundary[k] + (
                        x_ver_b[1] + x_ver_b[0]
                    ) / 2
                    y_G_ij_k = y_min
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append(
                        (x_ver_b[1] - x_ver_b[0]) / 2 * weight_G_boundary[k]
                    )

            if i == 1:  # right boundary
                y_ver_b = np.array(
                    [
                        y_min + (y_max - y_min) / n_intervals * j,
                        y_min + (y_max - y_min) / n_intervals * (j + 1),
                    ]
                )

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = x_max
                    y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_boundary[k] + (
                        y_ver_b[1] + y_ver_b[0]
                    ) / 2
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append(
                        (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_boundary[k]
                    )

                """
                since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
                for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jacobian term
                if we integral from left to right, ds = dx
                """
            if i == 2:  # top boundary
                x_ver_b = np.array(
                    [
                        x_min + (x_max - x_min) / n_intervals * j,
                        x_min + (x_max - x_min) / n_intervals * (j + 1),
                    ]
                )
                # if x_ver_b = np.array([x_max-(x_max-x_min)/n_intervals*j, x_max-(x_max-x_min)/n_intervals*(j+1)]), we integral from right to left, det_J_b_time_weight should be -((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = (x_ver_b[1] - x_ver_b[0]) / 2 * x_G_boundary[k] + (
                        x_ver_b[1] + x_ver_b[0]
                    ) / 2  # if
                    y_G_ij_k = y_max
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append(
                        (x_ver_b[1] - x_ver_b[0]) / 2 * weight_G_boundary[k]
                    )

                """
                since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
                for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
                """
            if i == 3:  # left boundary
                y_ver_b = np.array(
                    [
                        y_min + (y_max - y_min) / n_intervals * j,
                        y_min + (y_max - y_min) / n_intervals * (j + 1),
                    ]
                )
                # if y_ver_b = np.array([y_max-(y_max-y_min)/n_intervals*j, y_max-(y_max-y_min)/n_intervals*(j+1)]), we integral from top to bottom, det_J_b_time_weight should be -((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = x_min
                    y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_boundary[k] + (
                        y_ver_b[1] + y_ver_b[0]
                    ) / 2
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append(
                        (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_boundary[k]
                    )
    return x_G_b, det_J_b_time_weight


@njit
def x_G_and_def_J_time_weight_multi_grains(
    num_of_cell,
    x_G_domain_rec,
    x_G_domain_tri,
    weight_G_domain_rec,
    weight_G_domain_tri,
    cell_shape,
    cell_nodes_list,
    grain_id,
    angle,
    repeated_vertex,
):
    gauss_angle = []  # corresponding angle of each gauss point
    x_G = []  # xy coordinates of gauss points in domain
    Gauss_grain_id = []
    det_J_time_weight = []  # determin of jacobian
    for i in range(num_of_cell):

        if cell_shape[i] == "rec":

            # in the ith cell calculate get xy coordinates of each domain vertex
            x_ver_mn = np.array(
                [
                    cell_nodes_list[i][0][0],
                    cell_nodes_list[i][1][0],
                    cell_nodes_list[i][2][0],
                    cell_nodes_list[i][3][0],
                ],
                dtype=np.float64,
            )
            y_ver_mn = np.array(
                [
                    cell_nodes_list[i][0][1],
                    cell_nodes_list[i][1][1],
                    cell_nodes_list[i][2][1],
                    cell_nodes_list[i][3][1],
                ],
                dtype=np.float64,
            )
            # calculate the cy coordinates of gauss points in current integration domain
            for k in range(len(x_G_domain_rec)):
                gauss_angle.append(angle[angle.index(int(grain_id[i])) + 1])
                x_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                y_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                x_G.append([x_G_mn_k, y_G_mn_k])
                Gauss_grain_id.append(grain_id[i])
                J1 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][1]),
                                (-1 - x_G_domain_rec[k][1]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J2 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][1]),
                                (-1 - x_G_domain_rec[k][1]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                J3 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][0]),
                                (-1 - x_G_domain_rec[k][0]),
                                (1 + x_G_domain_rec[k][0]),
                                (1 - x_G_domain_rec[k][0]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J4 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][0]),
                                (-1 - x_G_domain_rec[k][0]),
                                (1 + x_G_domain_rec[k][0]),
                                (1 - x_G_domain_rec[k][0]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )

                det_J_time_weight.append(
                    np.linalg.det(np.array([[J1, J2], [J3, J4]]))
                    * weight_G_domain_rec[k]
                )  # each gauss point belongs to same cell has same jacobian

        if cell_shape[i] == "tri":

            # # in the ith cell calculate get xy coordinates of each domain vertex
            # x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0]],dtype=np.float64)
            # y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1]],dtype=np.float64)
            # # calculate the cy coordinates of gauss points in current integration domain
            # for k in range(len(x_G_domain_tri)):
            #     gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
            #     x_G_mn_k = cell_nodes_list[i][0][0] + (cell_nodes_list[i][1][0]-cell_nodes_list[i][0][0])*x_G_domain_tri[k][0] + (cell_nodes_list[i][2][0]-cell_nodes_list[i][0][0])*x_G_domain_tri[k][1]
            #     y_G_mn_k = cell_nodes_list[i][0][1] + (cell_nodes_list[i][1][1]-cell_nodes_list[i][0][1])*x_G_domain_tri[k][0] + (cell_nodes_list[i][2][1]-cell_nodes_list[i][0][1])*x_G_domain_tri[k][1]
            #     x_G.append([x_G_mn_k, y_G_mn_k])
            #     J1 = cell_nodes_list[i][1][0] - cell_nodes_list[i][0][0]
            #     J2 = cell_nodes_list[i][1][1] - cell_nodes_list[i][0][1]
            #     J3 = cell_nodes_list[i][2][0] - cell_nodes_list[i][0][0]
            #     J4 = cell_nodes_list[i][2][1] - cell_nodes_list[i][0][1]

            #     det_J_time_weight.append(np.linalg.det(np.array([[J1, J3],[J2,J4]]))*weight_G_domain_tri[k])  # each gauss point belongs to same cell has same jacobian

            # in the ith cell calculate get xy coordinates of each domain vertex
            if repeated_vertex[i] == "first":
                x_ver_mn = np.array(
                    [
                        cell_nodes_list[i][0][0],
                        cell_nodes_list[i][0][0],
                        cell_nodes_list[i][1][0],
                        cell_nodes_list[i][2][0],
                    ],
                    dtype=np.float64,
                )
                y_ver_mn = np.array(
                    [
                        cell_nodes_list[i][0][1],
                        cell_nodes_list[i][0][1],
                        cell_nodes_list[i][1][1],
                        cell_nodes_list[i][2][1],
                    ],
                    dtype=np.float64,
                )
            if repeated_vertex[i] == "three":
                x_ver_mn = np.array(
                    [
                        cell_nodes_list[i][0][0],
                        cell_nodes_list[i][1][0],
                        cell_nodes_list[i][2][0],
                        cell_nodes_list[i][2][0],
                    ],
                    dtype=np.float64,
                )
                y_ver_mn = np.array(
                    [
                        cell_nodes_list[i][0][1],
                        cell_nodes_list[i][1][1],
                        cell_nodes_list[i][2][1],
                        cell_nodes_list[i][2][1],
                    ],
                    dtype=np.float64,
                )
            # calculate the cy coordinates of gauss points in current integration domain
            for k in range(len(x_G_domain_rec)):
                gauss_angle.append(angle[angle.index(int(grain_id[i])) + 1])
                x_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                y_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][0]) * (1 + x_G_domain_rec[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                x_G.append([x_G_mn_k, y_G_mn_k])
                Gauss_grain_id.append(grain_id[i])
                # if i == 3298:
                # print(x_G_mn_k, y_G_mn_k)
                J1 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][1]),
                                (-1 - x_G_domain_rec[k][1]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J2 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][1]),
                                (1 - x_G_domain_rec[k][1]),
                                (1 + x_G_domain_rec[k][1]),
                                (-1 - x_G_domain_rec[k][1]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                J3 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][0]),
                                (-1 - x_G_domain_rec[k][0]),
                                (1 + x_G_domain_rec[k][0]),
                                (1 - x_G_domain_rec[k][0]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J4 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain_rec[k][0]),
                                (-1 - x_G_domain_rec[k][0]),
                                (1 + x_G_domain_rec[k][0]),
                                (1 - x_G_domain_rec[k][0]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                # det_J_time_weight.append(5.0e-15)
                det_J_time_weight.append(
                    np.linalg.det(np.array([[J1, J2], [J3, J4]]))
                    * weight_G_domain_rec[k]
                )  # each gauss point belongs to same cell has same jacobian
    return x_G, det_J_time_weight, gauss_angle, Gauss_grain_id


# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian on boundaries


@njit
def x_G_b_and_det_J_b_multi_grains(
    x_min,
    x_max,
    y_min,
    y_max,
    bottom_boundary_cell_nodes_list,
    right_boundary_cell_nodes_list,
    top_boundary_cell_nodes_list,
    left_boundary_cell_nodes_list,
    x_G_boundary,
    weight_G_boundary,
    grain_id_bottom,
    grain_id_top,
    grain_id_left,
    grain_id_right,
    angle,
):
    gauss_angle_b = []
    x_G_b = []

    det_J_b_time_weight = []  # determin of jacobian
    Gauss_b_grain_id = []

    for j in range(
        len(bottom_boundary_cell_nodes_list)
    ):  # the jth interval on ith bnoundary
        x_ver_b = np.array(
            [
                bottom_boundary_cell_nodes_list[j][0],
                bottom_boundary_cell_nodes_list[j][1],
            ]
        )

        for k in range(len(x_G_boundary)):
            x_G_ij_k = (x_ver_b[1] - x_ver_b[0]) / 2 * x_G_boundary[k] + (
                x_ver_b[1] + x_ver_b[0]
            ) / 2
            y_G_ij_k = y_min
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            gauss_angle_b.append(angle[angle.index(grain_id_bottom[j]) + 1])
            Gauss_b_grain_id.append(grain_id_bottom[j])

            det_J_b_time_weight.append(
                (x_ver_b[1] - x_ver_b[0]) / 2 * weight_G_boundary[k]
            )

    for j in range(
        len(right_boundary_cell_nodes_list)
    ):  # the jth interval on ith bnoundary
        y_ver_b = np.array(
            [right_boundary_cell_nodes_list[j][0], right_boundary_cell_nodes_list[j][1]]
        )

        for k in range(len(x_G_boundary)):
            x_G_ij_k = x_max
            y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_boundary[k] + (
                y_ver_b[1] + y_ver_b[0]
            ) / 2
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            Gauss_b_grain_id.append(grain_id_right[j])
            gauss_angle_b.append(angle[angle.index(grain_id_right[j]) + 1])
            det_J_b_time_weight.append(
                (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_boundary[k]
            )

    for j in range(
        len(top_boundary_cell_nodes_list)
    ):  # the jth interval on ith bnoundary
        x_ver_b = np.array(
            [top_boundary_cell_nodes_list[j][0], top_boundary_cell_nodes_list[j][1]]
        )

        for k in range(len(x_G_boundary)):
            x_G_ij_k = (x_ver_b[1] - x_ver_b[0]) / 2 * x_G_boundary[k] + (
                x_ver_b[1] + x_ver_b[0]
            ) / 2  # if
            y_G_ij_k = y_max
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            gauss_angle_b.append(angle[angle.index(grain_id_top[j]) + 1])
            Gauss_b_grain_id.append(grain_id_top[j])
            det_J_b_time_weight.append(
                (x_ver_b[1] - x_ver_b[0]) / 2 * weight_G_boundary[k]
            )

    for j in range(
        len(left_boundary_cell_nodes_list)
    ):  # the jth interval on ith bnoundary
        y_ver_b = np.array(
            [left_boundary_cell_nodes_list[j][0], left_boundary_cell_nodes_list[j][1]]
        )

        for k in range(len(x_G_boundary)):
            x_G_ij_k = x_min
            y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_boundary[k] + (
                y_ver_b[1] + y_ver_b[0]
            ) / 2
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            gauss_angle_b.append(angle[angle.index(grain_id_left[j]) + 1])
            Gauss_b_grain_id.append(grain_id_left[j])
            det_J_b_time_weight.append(
                (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_boundary[k]
            )

            """
            since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
            for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jaobian term
            if we integral from left to right, ds = dx
            """

            """
            since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
            for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
            """

    return x_G_b, det_J_b_time_weight, gauss_angle_b, Gauss_b_grain_id


# get all gauss points in domain, 2d domain
def x_G_and_def_J_time_weight_2d_fuelcell_domain(
    cell_nodes_x, cell_nodes_y, x_G_domain, weight_G_domain
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian

    for i in range(np.shape(cell_nodes_x)[0]):
        # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
        x_ver_mn = cell_nodes_x[i, :]
        y_ver_mn = cell_nodes_y[i, :]
        # calculate the cy coordinates of gauss points in current integration domain
        for k in range(len(x_G_domain)):

            x_G_mn_k = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            y_G_mn_k = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(y_ver_mn),
                )
            )

            x_G.append([x_G_mn_k, y_G_mn_k])

            J1 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][1]),
                            (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][1]),
                            (-1 - x_G_domain[k][1]),
                        ]
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J2 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][1]),
                            (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][1]),
                            (-1 - x_G_domain[k][1]),
                        ]
                    ),
                    np.transpose(y_ver_mn),
                )
            )
            J3 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][0]),
                            (-1 - x_G_domain[k][0]),
                            (1 + x_G_domain[k][0]),
                            (1 - x_G_domain[k][0]),
                        ]
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J4 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][0]),
                            (-1 - x_G_domain[k][0]),
                            (1 + x_G_domain[k][0]),
                            (1 - x_G_domain[k][0]),
                        ]
                    ),
                    np.transpose(y_ver_mn),
                )
            )

            det_J_time_weight.append(
                np.linalg.det(np.array([[J1, J2], [J3, J4]])) * weight_G_domain[k]
            )

    return x_G, det_J_time_weight


# 2D dmain, 1d line boundary
@jit
def x_G_and_def_J_time_weight_2d_fuelcell_boundary(
    cell_nodes, x_coord, x_G_line, weight_G_line
):

    x_G_b_line = []
    det_J_b_time_weight_line = []  # determin of jacobian

    for i in range(np.shape(cell_nodes)[0]):
        y_ver1 = cell_nodes[i, 0]
        y_ver2 = cell_nodes[i, 1]

        for k in range(len(x_G_line)):
            y_G_ij_k = (y_ver2 - y_ver1) / 2 * x_G_line[k] + (y_ver2 + y_ver1) / 2
            x_G_ij_k = x_coord
            x_G_b_line.append([x_G_ij_k, y_G_ij_k])

            det_J_b_time_weight_line.append((y_ver2 - y_ver1) / 2 * weight_G_line[k])

    return x_G_b_line, det_J_b_time_weight_line


# get all gauss points in domain, 3d domain
def x_G_and_def_J_time_weight_3d_fuelcell_domain(
    cell_nodes_x, cell_nodes_y, cell_nodes_z, x_G_domain, weight_G_domain
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian
    for i in range(np.shape(cell_nodes_x)[0]):
        # in the mnnm (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
        x_ver_mn = cell_nodes_x[i, :]
        y_ver_mn = cell_nodes_y[i, :]
        z_ver_mn = cell_nodes_z[i, :]
        # calculate the cy coordinates of gauss points in current integration domain
        for k in range(len(x_G_domain)):

            x_G_mn_k = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            (1 + x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            y_G_mn_k = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            (1 + x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(y_ver_mn),
                )
            )
            z_G_mn_k = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            (1 + x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 + x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0])
                            * (1 - x_G_domain[k][1])
                            * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(z_ver_mn),
                )
            )

            x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

            J11 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            -(1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            -(1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            -(1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            -(1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J12 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J13 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J21 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            -(1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            -(1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            -(1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            -(1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(y_ver_mn),
                )
            )
            J22 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(y_ver_mn),
                )
            )
            J23 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(y_ver_mn),
                )
            )
            J31 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            -(1 + x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            -(1 - x_G_domain[k][1]) * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            -(1 + x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                            -(1 - x_G_domain[k][1]) * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(z_ver_mn),
                )
            )
            J32 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][2]),
                            -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                            -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][2]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(z_ver_mn),
                )
            )
            J33 = (
                1.0
                / 8.0
                * np.dot(
                    np.array(
                        [
                            -(1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            -(1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            -(1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            -(1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(z_ver_mn),
                )
            )

            det_J_time_weight.append(
                np.linalg.det(
                    np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])
                )
                * weight_G_domain[k]
            )

    return x_G, det_J_time_weight


# get all gauss points in domain, 3d domain, 2d boundary
def x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
    cell_nodes_boundary_x,
    cell_nodes_boundary_z,
    y_coords_on_boundary,
    x_G_domain,
    weight_G_domain,
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian

    for i in range(np.shape(cell_nodes_boundary_x)[0]):
        # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
        x_ver_mn = cell_nodes_boundary_x[i, :]
        z_ver_mn = cell_nodes_boundary_z[i, :]
        # calculate the cy coordinates of gauss points in current integration domain
        for k in range(len(x_G_domain)):

            x_G_mn_k = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            z_G_mn_k = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                        ],
                        dtype=np.float64,
                    ),
                    np.transpose(z_ver_mn),
                )
            )
            y_G_mn_k = y_coords_on_boundary

            x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

            J1 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][1]),
                            (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][1]),
                            (-1 - x_G_domain[k][1]),
                        ]
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J2 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][1]),
                            (1 - x_G_domain[k][1]),
                            (1 + x_G_domain[k][1]),
                            (-1 - x_G_domain[k][1]),
                        ]
                    ),
                    np.transpose(z_ver_mn),
                )
            )
            J3 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][0]),
                            (-1 - x_G_domain[k][0]),
                            (1 + x_G_domain[k][0]),
                            (1 - x_G_domain[k][0]),
                        ]
                    ),
                    np.transpose(x_ver_mn),
                )
            )
            J4 = (
                1.0
                / 4.0
                * np.dot(
                    np.array(
                        [
                            -(1 - x_G_domain[k][0]),
                            (-1 - x_G_domain[k][0]),
                            (1 + x_G_domain[k][0]),
                            (1 - x_G_domain[k][0]),
                        ]
                    ),
                    np.transpose(z_ver_mn),
                )
            )

            det_J_time_weight.append(
                np.linalg.det(np.array([[J1, J2], [J3, J4]])) * weight_G_domain[k]
            )

    return x_G, det_J_time_weight


def x_G_b_and_det_J_b_time_weight_2d_fuelcell_boundary(
    cell_nodes_boundary_x, cell_nodes_boundary_y, x_G_line, weight_G_line
):
    x_G_b_line = []
    det_J_b_time_weight_line = []  # determin of jacobian

    for i in range(np.shape(cell_nodes_boundary_x)[0]):
        x_ver1 = cell_nodes_boundary_x[i, 0]
        y_ver1 = cell_nodes_boundary_y[i, 0]
        x_ver2 = cell_nodes_boundary_x[i, 1]
        y_ver2 = cell_nodes_boundary_y[i, 1]

        if x_ver1 == x_ver2:
            y_ver_b = np.array([y_ver1, y_ver2])

            for k in range(len(x_G_line)):
                y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_line[k] + (
                    y_ver_b[1] + y_ver_b[0]
                ) / 2
                x_G_ij_k = x_ver1
                x_G_b_line.append([x_G_ij_k, y_G_ij_k])

                det_J_b_time_weight_line.append(
                    (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_line[k]
                )

        if y_ver1 == y_ver2:  # right boundary
            x_ver_b = np.array([x_ver1, x_ver2])

            for k in range(len(x_G_line)):
                y_G_ij_k = y_ver1
                x_G_ij_k = (x_ver_b[1] - x_ver_b[0]) / 2 * x_G_line[k] + (
                    x_ver_b[1] + x_ver_b[0]
                ) / 2
                x_G_b_line.append([x_G_ij_k, y_G_ij_k])

                det_J_b_time_weight_line.append(
                    (x_ver_b[1] - x_ver_b[0]) / 2 * weight_G_line[k]
                )

    return x_G_b_line, det_J_b_time_weight_line


def x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(
    cell_nodes_boundary_x,
    cell_nodes_boundary_y,
    cell_nodes_boundary_z,
    x_G_domain,
    weight_G_domain,
):
    x_G = []  # xy coordinates of gauss points in domain
    det_J_time_weight = []  # determin of jacobian

    for i in range(np.shape(cell_nodes_boundary_x)[0]):
        # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
        x_ver_mn = cell_nodes_boundary_x[i, :]
        y_ver_mn = cell_nodes_boundary_y[i, :]
        z_ver_mn = cell_nodes_boundary_z[i, :]
        # calculate the cy coordinates of gauss points in current integration domain
        for k in range(len(x_G_domain)):
            if (
                y_ver_mn[0] == y_ver_mn[1]
                and y_ver_mn[1] == y_ver_mn[2]
                and y_ver_mn[2] == y_ver_mn[3]
            ):

                x_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                z_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(z_ver_mn),
                    )
                )
                y_G_mn_k = y_ver_mn[0]

                x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

                J1 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J2 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(z_ver_mn),
                    )
                )
                J3 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J4 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(z_ver_mn),
                    )
                )

                det_J_time_weight.append(
                    np.linalg.det(np.array([[J1, J2], [J3, J4]])) * weight_G_domain[k]
                )
            if (
                x_ver_mn[0] == x_ver_mn[1]
                and x_ver_mn[1] == x_ver_mn[2]
                and x_ver_mn[2] == x_ver_mn[3]
            ):

                y_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                z_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(z_ver_mn),
                    )
                )
                x_G_mn_k = x_ver_mn[0]

                x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

                J1 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                J2 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(z_ver_mn),
                    )
                )
                J3 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                J4 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(z_ver_mn),
                    )
                )

                det_J_time_weight.append(
                    np.linalg.det(np.array([[J1, J2], [J3, J4]])) * weight_G_domain[k]
                )
            if (
                z_ver_mn[0] == z_ver_mn[1]
                and z_ver_mn[1] == z_ver_mn[2]
                and z_ver_mn[2] == z_ver_mn[3]
            ):

                x_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                y_G_mn_k = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                (1 - x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                                (1 - x_G_domain[k][0]) * (1 + x_G_domain[k][1]),
                            ],
                            dtype=np.float64,
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                z_G_mn_k = z_ver_mn[0]

                x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

                J1 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J2 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][1]),
                                (1 - x_G_domain[k][1]),
                                (1 + x_G_domain[k][1]),
                                (-1 - x_G_domain[k][1]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )
                J3 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(x_ver_mn),
                    )
                )
                J4 = (
                    1.0
                    / 4.0
                    * np.dot(
                        np.array(
                            [
                                -(1 - x_G_domain[k][0]),
                                (-1 - x_G_domain[k][0]),
                                (1 + x_G_domain[k][0]),
                                (1 - x_G_domain[k][0]),
                            ]
                        ),
                        np.transpose(y_ver_mn),
                    )
                )

                det_J_time_weight.append(
                    np.linalg.det(np.array([[J1, J2], [J3, J4]])) * weight_G_domain[k]
                )

    return x_G, det_J_time_weight


# 3D dmain, 1d line boundary
@jit
def x_G_and_det_J_line_3d_fuelcell_1d_boundary(
    segments_source, x_G_line, weight_G_line
):

    x_G_b_line = []
    det_J_b_time_weight_line = []  # determin of jacobian

    for i in range(np.shape(segments_source)[0]):
        x_ver1 = segments_source[i, 0]
        y_ver1 = segments_source[i, 1]
        z_ver1 = segments_source[i, 2]
        x_ver2 = segments_source[i, 3]
        y_ver2 = segments_source[i, 4]
        z_ver2 = segments_source[i, 5]

        if x_ver1 == x_ver2 and z_ver1 == z_ver2:
            y_ver_b = np.array([y_ver1, y_ver2])

            for k in range(len(x_G_line)):
                y_G_ij_k = (y_ver_b[1] - y_ver_b[0]) / 2 * x_G_line[k] + (
                    y_ver_b[1] + y_ver_b[0]
                ) / 2
                z_G_ij_k = z_ver1
                x_G_ij_k = x_ver1
                x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                det_J_b_time_weight_line.append(
                    (y_ver_b[1] - y_ver_b[0]) / 2 * weight_G_line[k]
                )

        if x_ver1 == x_ver2 and y_ver1 == y_ver2:  # right boundary
            z_ver_b = np.array([z_ver1, z_ver2])

            for k in range(len(x_G_line)):
                x_G_ij_k = x_ver1
                y_G_ij_k = y_ver1
                z_G_ij_k = (z_ver_b[1] - z_ver_b[0]) / 2 * x_G_line[k] + (
                    z_ver_b[1] + z_ver_b[0]
                ) / 2
                x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                det_J_b_time_weight_line.append(
                    (z_ver_b[1] - z_ver_b[0]) / 2 * weight_G_line[k]
                )

        if z_ver1 == z_ver2 and y_ver1 == y_ver2:  # right boundary
            x_ver_b = np.array([x_ver1, x_ver2])

            for k in range(len(x_G_line)):
                z_G_ij_k = z_ver1
                y_G_ij_k = y_ver1
                x_G_ij_k = (x_ver_b[1] - x_ver_b[0]) / 2 * x_G_line[k] + (
                    x_ver_b[1] + x_ver_b[0]
                ) / 2
                x_G_b_line.append([x_G_ij_k, y_G_ij_k, z_G_ij_k])

                det_J_b_time_weight_line.append(
                    (x_ver_b[1] - x_ver_b[0]) / 2 * weight_G_line[k]
                )

    return x_G_b_line, det_J_b_time_weight_line
