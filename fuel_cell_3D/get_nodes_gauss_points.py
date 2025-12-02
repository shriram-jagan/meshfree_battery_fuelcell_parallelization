import numpy as np
from numba import jit


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

    cell_nodes_fixed_x = []  # n by 4.
    cell_nodes_fixed_z = []

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
                    cell_nodes_fixed_x.append(
                        [
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                            x_min + (x_max - x_min) / (num_pixels_x) * i,
                        ]
                    )
                    cell_nodes_fixed_z.append(
                        [
                            z_min,
                            z_min,
                            z_min + (z_max - z_min) / (num_pixels_z),
                            z_min + (z_max - z_min) / (num_pixels_z),
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
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
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
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i + 1),
                                x_min + (x_max - x_min) / (num_pixels_x) * (i),
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
        cell_nodes_fixed_x,
        cell_nodes_fixed_z,
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
