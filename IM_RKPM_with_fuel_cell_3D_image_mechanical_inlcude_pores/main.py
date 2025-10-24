import time

start_time = time.time()
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from define_buttler_volmer import (
    Dn_complex,
    alpha_lattice_complex,
    c_lattice_complex,
    i_0_complex,
    i_se,
    ocp_complex,
)
from define_diffusion_matrix_form import (
    diffusion_matrix,
    diffusion_matrix_fuel_cell,
    diffusion_matrix_fuel_cell_distributed_point_source,
)
from define_eva_at_gauss_points import evaluate_at_gauss_points
from define_mechanical_stiffness_matrix import (
    mechanical_C_tensor,
    mechanical_C_tensor_3d,
    mechanical_force_matrix,
    mechanical_force_matrix_3d,
    mechanical_stiffness_matrix_3d_fuel_cell,
    mechanical_stiffness_matrix_battery,
    mechanical_stiffness_matrix_fuel_cell,
)
from get_nodes_gauss_points import (
    get_x_nodes_fuel_cell_2d_toy,
    get_x_nodes_fuel_cell_2d_toy_image,
    get_x_nodes_fuel_cell_3d_toy,
    get_x_nodes_fuel_cell_3d_toy_image,
    get_x_nodes_multi_grain,
    get_x_nodes_single_grain_battery,
    x_G_and_def_J_time_weight_2d_fuelcell_boundary,
    x_G_and_def_J_time_weight_2d_fuelcell_domain,
    x_G_and_def_J_time_weight_3d_fuelcell_domain,
    x_G_and_def_J_time_weight_multi_grains,
    x_G_and_def_J_time_weight_structured,
    x_G_and_det_J_line_3d_fuelcell_1d_boundary,
    x_G_b_and_det_J_b_multi_grains,
    x_G_b_and_det_J_b_structured,
    x_G_b_and_det_J_b_time_weight_2d_fuelcell_boundary,
    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary,
    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface,
)
from numba import jit, njit, typed
from numpy import sign
from numpy.linalg import eig, norm
from read_image import read_in_image
from scipy.sparse import block_diag, bmat, csc_matrix, csr_matrix, vstack
from scipy.sparse.linalg import eigs, inv, spsolve
from shape_function_in_domain import (
    compute_phi_M,
    shape_func_n_nodes_by_n_nodes,
    shape_grad_shape_func,
)
from tqdm import tqdm

###################################
# define geometry and analysis type
###################################
print("define geometry and analysis types")

single_grain = "False"  # True: single grain, False: read from an image
dimention = 3  # 3d or 2d

IM_RKPM = "False"  # if it is interfacial modified RKPM, only available for battery
studied_physics = "fuel cell"  # fuel cell or battery
damage_model = "OFF"  # ON or OFF

delta_point_source = "True"  # if point source is delta function. If the point source is distributed, set it to be 'False'

#########################################
# define differential and integral method
#########################################
differential_method = "direct"  # 'implicite' or 'direct'    # specify which differential method to use, implicite: H1, H2, direct: directly differentiate
# if the IM_RKPM=True, differential method must be set to be direct.
integral_method = "gauss"

################
# Define domain
################
print("Define domain and parameters")

if single_grain == "True":
    n_intervals = 10  # number of intervals along each direction
    n_nodes = n_intervals + 1  # number of nodes along each direction,
    # if from readin image, the intervals along each direction would be the same as number of pixel/voxel in this direction.

if studied_physics == "battery":
    if dimention == 2:
        x_min = -10e-6
        x_max = 10e-6
        y_min = -10e-6
        y_max = 10e-6
    if dimention == 3:
        x_min = -10e-6
        x_max = 10e-6
        y_min = -10e-6
        y_max = 10e-6
        z_min = -10e-6
        z_max = 10e-6

# for fuel cell:
#               if 2d: electrolyte is at left (small x) while electrode is at right (bigger x)
#               if 3d: electrolyte is at left (small y) while electrode is at right (bigger y)
# if this is changed, you need to update the normal vector:
# normal_vector_x_electrolyte
# normal_vector_x_electrode

# normal_vector_y_electrode
# normal_vector_y_electrolyte

# normal_vector_z_electrode
# normal_vector_z_electrolyte

if studied_physics == "fuel cell":
    if single_grain == "True":
        if dimention == 2:
            x_min_electrolyte = -10e-6
            x_max_electrolyte = 0.0
            y_min_electrolyte = 0.0
            y_max_electrolyte = 10e-6
            x_min_electrode = 0.0
            x_max_electrode = 10e-6
            y_min_electrode = 0.0
            y_max_electrode = 10e-6

        if dimention == 3:
            x_min_electrolyte = 0.0
            x_max_electrolyte = 10e-6
            y_min_electrolyte = -10e-6
            y_max_electrolyte = 0.0
            z_min_electrolyte = 0.0
            z_max_electrolyte = 10e-6
            x_min_electrode = 0.0
            x_max_electrode = 10e-6
            y_min_electrode = 0.0
            y_max_electrode = 10e-6
            z_min_electrode = 0.0
            z_max_electrode = 10e-6

    else:  # image based
        # total domain from readin image
        if dimention == 2:
            x_min = 0
            x_max = 20e-6
            y_min = 0
            y_max = 10e-6

        if dimention == 3:
            x_min = 0
            x_max = 20e-6
            y_min = 0
            y_max = 40e-6
            z_min = 0
            z_max = 20e-6

###############################
# Define time step
###############################
print("define time step")
# this is only used for battery simulation as we are solving the steady state for fuel cell.
if studied_physics == "battery":
    t = 100.0  # simulate for 10s
    nt = 1000  # nt is the number of time steps
    dt = t / nt  # time step

##############
# grain angle
##############
print("define grain angle")

if single_grain == "False" and studied_physics == "battery":
    angle = [
        26.0,
        np.pi,
        75.0,
        np.pi / 4.0,
        121.0,
        np.pi * 2.0 / 3.0,
        149.0,
        np.pi / 2.0,
        90.0,
        np.pi / 3.0,
        81.0,
        np.pi / 4.0,
        37.0,
        np.pi * 2.0 / 3.0,
        110.0,
        0.0,
    ]
else:
    angle = 0

###############################
# Define material properties
###############################

Fday = 9.6485e4  # Faraday constant
R = 8.3145e0  # gas constant

# for damage
k_i = 0.0125
k_f = 0.015

if studied_physics == "battery":
    # for diffusion
    Tk = 3.0515e2  # temperature in K
    c_max = 49600.0  # maximum concentration
    k_con = 10.0  # conductivity
    Dx_div_Dy = 100.0
    j_applied = -15.0  # j_applied

    # for mechanical
    E = 138.87e9  # Youngs modulus (Pa)
    nu = 0.3  # Poisson ratio
    lambda_mechanical = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)  # lamme constants

else:
    diffusion_electrolyte = 0.035
    diffusion_electrode = 1.0e-9
    diffusion_pore = 1.0e-7

    k_gas = 1.0e-6

    i_0 = 1.0e-3
    i_0_solid = 1.0e-1

    T = 1273.2

    E_0 = 1.0

    V_app = 1.5

    c_boundary = 1000.0

    c_boundary_pore = 9.572

    # for mechanical
    E_electrolyte = 133.0e9  # Youngs modulus (Pa)
    nu_electrolyte = 0.33  # Poisson ratio
    lambda_mechanical_electrolyte = (
        E_electrolyte * nu_electrolyte / (1 + nu_electrolyte) / (1 - 2 * nu_electrolyte)
    )
    mu_electrolyte = E_electrolyte / 2 / (1 + nu_electrolyte)  # lamme constants

    E_electrode = 130.0e9  # Youngs modulus (Pa)
    nu_electrode = 0.33  # Poisson ratio
    lambda_mechanical_electrode = (
        E_electrode * nu_electrode / (1 + nu_electrode) / (1 - 2 * nu_electrode)
    )
    mu_electrode = E_electrode / 2 / (1 + nu_electrode)  # lamme constants

    beta_fuelcell_expansion_coefficient = 4.0e-6  # m^3/mol

######################
# Gauss integral
######################

if integral_method == "gauss":
    # Define Guass int points and weights

    # 3d cube:
    x_G_cube = [
        [-(3**0.5) / 3, -(3**0.5) / 3, -(3**0.5) / 3],
        [3**0.5 / 3, -(3**0.5) / 3, -(3**0.5) / 3],
        [-(3**0.5) / 3, 3**0.5 / 3, -(3**0.5) / 3],
        [3**0.5 / 3, 3**0.5 / 3, -(3**0.5) / 3],
        [-(3**0.5) / 3, -(3**0.5) / 3, 3**0.5 / 3],
        [3**0.5 / 3, -(3**0.5) / 3, 3**0.5 / 3],
        [-(3**0.5) / 3, 3**0.5 / 3, 3**0.5 / 3],
        [3**0.5 / 3, 3**0.5 / 3, 3**0.5 / 3],
    ]  # coordinates of 2D Gauss points in Neutral coordinate system for square doamin
    weight_G_cube = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # weight of each 2D Gauss points for rectangular

    # 2d rectangle or triangle
    x_G_rec = [
        [-(3**0.5) / 3, -(3**0.5) / 3],
        [-(3**0.5) / 3, 3**0.5 / 3],
        [3**0.5 / 3, -(3**0.5) / 3],
        [3**0.5 / 3, 3**0.5 / 3],
    ]  # coordinates of 2D Gauss points in Neutral coordinate system for square doamin
    x_G_tri = [[1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0]]
    weight_G_rec = [
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # weight of each 2D Gauss points for rectangular
    weight_G_tri = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    # 1d line
    x_G_line = [
        -((3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
        -((3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
        (3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
        (3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
    ]  # [-0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0,0.4058451513773972,0.7415311855993945,0.9491079123427585]#                   # coordinates of 1D Gauss points
    weight_G_line = [
        0.5 - 30**0.5 / 36,
        0.5 + 30**0.5 / 36,
        0.5 + 30**0.5 / 36,
        0.5 - 30**0.5 / 36,
    ]  # [0.1294849661688697,0.2797053914892766,0.3818300505051189,0.4179591836734694,0.3818300505051189,0.2797053914892766,0.1294849661688697]#         # weight of each 1D Gauss points

def_para_time = time.time()

print("time to define parameters = " + "%s seconds" % (def_para_time - start_time))

##################
# define RK nodes
##################
print("define RK nodes")

if studied_physics == "battery" and single_grain == "True":

    x_nodes = np.array(
        get_x_nodes_single_grain_battery(
            n_nodes, n_intervals, x_min, x_max, y_min, y_max
        )
    )  # types: array

    num_interface_segments = 0
    interface_nodes = np.zeros((1, 1))
    BxByCxCy = np.zeros((1, 1))

    num_nodes = np.shape(x_nodes)[0]
    print("number of nodes: " + str(num_nodes))

    nodes_grain_id = 1 * np.ones(num_nodes)

if studied_physics == "battery" and single_grain == "False":
    file_name = "reduced_8grain_ulm_large_grain_51x51.tiff"
    img_, unic_grain_id, num_pixels_xyz = read_in_image(
        file_name, studied_physics, dimention
    )
    (
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
    ) = get_x_nodes_multi_grain(x_min, x_max, y_min, y_max, num_pixels_xyz, img_)
    num_interface_segments = np.shape(interface_segments)[0]
    interface_nodes = np.asarray(interface_segments).reshape(
        num_interface_segments * 2, 2
    )
    BxByCxCy = np.asarray(interface_segments).reshape(
        num_interface_segments, 4
    )  # the first column is x coordinates of point B, ......

    # cell_nodes_list: the ith row correspondes to ith domain cell, it includes the coordinates of all nodes that forms this cell
    # grain_id: grain id of each domain cell
    # grain_id_bottom: grain id of bottom boundary cell
    # cell_shape: triangle or rectangle domain cell
    # num_rec_cell: total number of rectangle cell
    # x_nodes: coordinates of all nodes, shape: number of nodes times 2
    # bottom_boundary_cell_nodes_list: coordinates
    # print(len(repeated_vertex))

    boundary_nodes = np.asarray(
        bottom_boundary_cell_nodes_list
        + top_boundary_cell_nodes_list
        + left_boundary_cell_nodes_list
        + right_boundary_cell_nodes_list
    )

    # x_nodes is array, all others are lists
    num_of_cell = int(num_rec_cell + num_tri_cell)

    x_nodes = np.array(x_nodes)

    num_nodes = np.shape(x_nodes)[0]
    print("number of nodes: " + str(num_nodes))

if studied_physics == "fuel cell":

    if single_grain == "True":
        if dimention == 2:
            (
                x_nodes_mechanical,
                x_nodes_electrolyte,
                x_nodes_electrode,
                point_source_coords,
                fixed_nodes_coords,
                nodes_id_left_electrolyte,
                nodes_id_right_electrode,
                cell_nodes_electrolyte_x,
                cell_nodes_electrolyte_y,
                cell_nodes_electrode_x,
                cell_nodes_electrode_y,
                cell_nodes_left_electrolyte_y,
                cell_nodes_right_electrode_y,
            ) = get_x_nodes_fuel_cell_2d_toy(
                x_min_electrolyte,
                x_max_electrolyte,
                y_min_electrolyte,
                y_max_electrolyte,
                x_min_electrode,
                x_max_electrode,
                y_min_electrode,
                y_max_electrode,
                n_intervals,
            )
            (
                x_nodes_mechanical,
                x_nodes_electrolyte,
                x_nodes_electrode,
                point_source_coords,
                fixed_nodes_coords,
                nodes_id_left_electrolyte,
                nodes_id_right_electrode,
                cell_nodes_electrolyte_x,
                cell_nodes_electrolyte_y,
                cell_nodes_electrode_x,
                cell_nodes_electrode_y,
                cell_nodes_left_electrolyte_y,
                cell_nodes_right_electrode_y,
            ) = [
                np.asarray(lst)
                for lst in [
                    x_nodes_mechanical,
                    x_nodes_electrolyte,
                    x_nodes_electrode,
                    point_source_coords,
                    fixed_nodes_coords,
                    nodes_id_left_electrolyte,
                    nodes_id_right_electrode,
                    cell_nodes_electrolyte_x,
                    cell_nodes_electrolyte_y,
                    cell_nodes_electrode_x,
                    cell_nodes_electrode_y,
                    cell_nodes_left_electrolyte_y,
                    cell_nodes_right_electrode_y,
                ]
            ]
        if dimention == 3:
            (
                x_nodes_mechanical,
                x_nodes_electrolyte,
                x_nodes_electrode,
                segments_source_coords,
                segments_fixed_coods,
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
            ) = get_x_nodes_fuel_cell_3d_toy(
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
            )
            (
                x_nodes_mechanical,
                x_nodes_electrolyte,
                x_nodes_electrode,
                segments_source_coords,
                segments_fixed_coods,
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
            ) = [
                np.asarray(lst)
                for lst in [
                    x_nodes_mechanical,
                    x_nodes_electrolyte,
                    x_nodes_electrode,
                    segments_source_coords,
                    segments_fixed_coods,
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
                ]
            ]

    if single_grain == "False":
        if dimention == 2:
            file_name = "M_2d_3phases_simple.tif"
            img_, unic_grain_id, num_pixels_xyz = read_in_image(
                file_name, studied_physics, dimention
            )

            (
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
            ) = get_x_nodes_fuel_cell_2d_toy_image(
                x_min, x_max, y_min, y_max, num_pixels_xyz, img_
            )

            (
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
            ) = [
                np.asarray(lst)
                for lst in [
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
                ]
            ]

            if delta_point_source == "False":
                cell_nodes_distributed_point_source_line_x = []
                cell_nodes_distributed_point_source_line_y = []
                # if 50 volxels on top right edge of electrolyte, 20% is used to distribute the point source, this is 10 cells
                for i_dis in range(10):
                    cell_nodes_distributed_point_source_line_x.append(
                        [(x_max + x_min) / 2, (x_max + x_min) / 2]
                    )
                    cell_nodes_distributed_point_source_line_y.append(
                        [
                            (y_max + y_min) / 2 + i_dis * (y_max - y_min) / 100,
                            (y_max + y_min) / 2 + (i_dis + 1) * (y_max - y_min) / 100,
                        ]
                    )

        if dimention == 3:
            # file_name = 'micro_3d_connected.tif'#'M_3d_3phases_simple.tif'# real geometry
            file_name = "M_3d_3phases_simple.tif"  # simple geometry
            img_, unic_grain_id, num_pixels_xyz = read_in_image(
                file_name, studied_physics, dimention
            )

            (
                x_nodes_mechanical,
                x_nodes_electrolyte,
                x_nodes_electrode,
                x_nodes_pore,
                segments_source_coords,
                segments_fixed_coods,
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
            ) = get_x_nodes_fuel_cell_3d_toy_image(
                x_min, x_max, y_min, y_max, z_min, z_max, num_pixels_xyz, img_
            )

            (
                x_nodes_mechanical,
                x_nodes_electrolyte,
                x_nodes_electrode,
                x_nodes_pore,
                segments_source_coords,
                segments_fixed_coods,
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
            ) = [
                np.asarray(lst)
                for lst in [
                    x_nodes_mechanical,
                    x_nodes_electrolyte,
                    x_nodes_electrode,
                    x_nodes_pore,
                    segments_source_coords,
                    segments_fixed_coods,
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
                ]
            ]

            if delta_point_source == "False":
                cell_nodes_distributed_point_source_surface_x = []
                cell_nodes_distributed_point_source_surface_y = []
                cell_nodes_distributed_point_source_surface_z = []
                # if 10 volxels on top right surface of electrolyte, 20% is used to distribute the point source, this is 2 cells
                for i_dis in range(2):
                    for j_dis in range(20):
                        cell_nodes_distributed_point_source_surface_x.append(
                            [
                                x_min + (x_max - x_min) / (20) * j_dis,
                                x_min + (x_max - x_min) / (20) * (j_dis + 1),
                                x_min + (x_max - x_min) / (20) * (j_dis + 1),
                                x_min + (x_max - x_min) / (20) * j_dis,
                            ]
                        )
                        cell_nodes_distributed_point_source_surface_y.append(
                            [
                                (y_max + y_min) / 2,
                                (y_max + y_min) / 2,
                                (y_max + y_min) / 2,
                                (y_max + y_min) / 2,
                            ]
                        )
                        cell_nodes_distributed_point_source_surface_z.append(
                            [
                                (z_max + z_min) / 2 + (z_max - z_min) / 20 * i_dis,
                                (z_max + z_min) / 2 + (z_max - z_min) / 20 * i_dis,
                                (z_max + z_min) / 2
                                + (z_max - z_min) / 20 * (i_dis + 1),
                                (z_max + z_min) / 2
                                + (z_max - z_min) / 20 * (i_dis + 1),
                            ]
                        )

    num_interface_segments = 0
    interface_nodes = np.zeros((1, 1))
    BxByCxCy = np.zeros((1, 1))

    num_nodes_electrolyte = np.shape(x_nodes_electrolyte)[0]
    num_nodes_electrode = np.shape(x_nodes_electrode)[0]
    num_nodes_pore = np.shape(x_nodes_pore)[0]
    num_nodes_mechanical = np.shape(x_nodes_mechanical)[0]

    nodes_grain_id_electrolyte = 1 * np.ones(num_nodes_electrolyte)
    nodes_grain_id_electrode = 1 * np.ones(num_nodes_electrode)
    nodes_grain_id_pore = 1 * np.ones(num_nodes_pore)

    nodes_grain_id_mechanical = 1 * np.ones(num_nodes_mechanical)

    print("number of nodes in electrolyte: " + str(num_nodes_electrolyte))
    print("number of nodes in electrode: " + str(num_nodes_electrode))
    print("number of nodes in pore: " + str(num_nodes_pore))
    print("number of nodes in whole domain: " + str(num_nodes_mechanical))

    print(
        "number of cells in electrolyte: " + str(np.shape(cell_nodes_electrolyte_x)[0])
    )
    print("number of cells in electrode: " + str(np.shape(cell_nodes_electrode_x)[0]))
    print("number of cells in pore: " + str(np.shape(cell_nodes_pore_x)[0]))
    # print('number of cells in whole domain: ' + str(np.shape(cell_nodes_mechanical_x)[0]))

# 9110 electrolyte cells
# 2939 electrode cells
# 3951 voids


##########################
# define gauss points
##########################
print("define gauss points")
# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian
if integral_method == "gauss":
    if studied_physics == "battery":
        if single_grain == "True":
            n_boundaries = 4
            x_G, det_J_time_weight = x_G_and_def_J_time_weight_structured(
                n_intervals, x_min, x_max, y_min, y_max, x_G_rec, weight_G_rec
            )
            x_G_b, det_J_b_time_weight = x_G_b_and_det_J_b_structured(
                n_boundaries,
                n_intervals,
                x_min,
                x_max,
                y_min,
                y_max,
                x_G_line,
                weight_G_line,
            )

            gauss_angle = angle * np.ones(len(x_G))
            gauss_angle_b = angle * np.ones(len(x_G_b))

            num_gauss_points_in_domain = np.shape(x_G)[0]
            num_gauss_points_on_boundary = np.shape(x_G_b)[0]
            Gauss_grain_id = 1 * np.ones(num_gauss_points_in_domain)
            Gauss_b_grain_id = 1 * np.ones(num_gauss_points_on_boundary)

        if single_grain == "False":
            x_G_domain_tri = typed.List([typed.List(x) for x in x_G_tri])
            x_G_domain_rec = typed.List([typed.List(x) for x in x_G_rec])
            cell_nodes_list = typed.List([typed.List(x) for x in cell_nodes_list])

            x_G, det_J_time_weight, gauss_angle, Gauss_grain_id = (
                x_G_and_def_J_time_weight_multi_grains(
                    num_of_cell,
                    x_G_domain_rec,
                    x_G_domain_tri,
                    weight_G_rec,
                    weight_G_tri,
                    cell_shape,
                    cell_nodes_list,
                    grain_id,
                    angle,
                    repeated_vertex,
                )
            )
            bottom_boundary_cell_nodes_list = typed.List(
                [typed.List(x) for x in bottom_boundary_cell_nodes_list]
            )
            right_boundary_cell_nodes_list = typed.List(
                [typed.List(x) for x in right_boundary_cell_nodes_list]
            )
            left_boundary_cell_nodes_list = typed.List(
                [typed.List(x) for x in left_boundary_cell_nodes_list]
            )
            top_boundary_cell_nodes_list = typed.List(
                [typed.List(x) for x in top_boundary_cell_nodes_list]
            )
            x_G_b, det_J_b_time_weight, gauss_angle_b, Gauss_b_grain_id = (
                x_G_b_and_det_J_b_multi_grains(
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    bottom_boundary_cell_nodes_list,
                    right_boundary_cell_nodes_list,
                    top_boundary_cell_nodes_list,
                    left_boundary_cell_nodes_list,
                    x_G_line,
                    weight_G_line,
                    grain_id_bottom,
                    grain_id_top,
                    grain_id_left,
                    grain_id_right,
                    angle,
                )
            )

        x_G = np.array(x_G)
        x_G_b = np.array(x_G_b)
        num_gauss_points_in_domain = np.shape(x_G)[0]
        num_gauss_points_on_boundary = np.shape(x_G_b)[0]
        gauss_angle = np.array(gauss_angle)
        gauss_angle_b = np.array(gauss_angle_b)
        Gauss_grain_id = np.array(Gauss_grain_id)
        Gauss_b_grain_id = np.array(Gauss_b_grain_id)

        print("number of Gauss points in domain: " + str(num_gauss_points_in_domain))
        print(
            "number of Gauss points on boundaries: " + str(num_gauss_points_on_boundary)
        )

    if studied_physics == "fuel cell":
        if dimention == 2:
            x_G_electrolyte, det_J_time_weight_electrolyte = (
                x_G_and_def_J_time_weight_2d_fuelcell_domain(
                    cell_nodes_electrolyte_x,
                    cell_nodes_electrolyte_y,
                    x_G_rec,
                    weight_G_rec,
                )
            )
            x_G_electrode, det_J_time_weight_electrode = (
                x_G_and_def_J_time_weight_2d_fuelcell_domain(
                    cell_nodes_electrode_x,
                    cell_nodes_electrode_y,
                    x_G_rec,
                    weight_G_rec,
                )
            )
            x_G_pore, det_J_time_weight_pore = (
                x_G_and_def_J_time_weight_2d_fuelcell_domain(
                    cell_nodes_pore_x, cell_nodes_pore_y, x_G_rec, weight_G_rec
                )
            )

            if single_grain == "True":
                x_G_b_electrolyte, det_J_b_time_weight_electrolyte = (
                    x_G_and_def_J_time_weight_2d_fuelcell_boundary(
                        cell_nodes_left_electrolyte_y,
                        x_min_electrolyte,
                        x_G_line,
                        weight_G_line,
                    )
                )
                x_G_b_electrode, det_J_b_time_weight_electrode = (
                    x_G_and_def_J_time_weight_2d_fuelcell_boundary(
                        cell_nodes_right_electrode_y,
                        x_max_electrode,
                        x_G_line,
                        weight_G_line,
                    )
                )
            else:
                x_G_b_electrolyte, det_J_b_time_weight_electrolyte = (
                    x_G_and_def_J_time_weight_2d_fuelcell_boundary(
                        cell_nodes_left_electrolyte_y, x_min, x_G_line, weight_G_line
                    )
                )
                x_G_b_electrode, det_J_b_time_weight_electrode = (
                    x_G_and_def_J_time_weight_2d_fuelcell_boundary(
                        cell_nodes_right_electrode_y, x_max, x_G_line, weight_G_line
                    )
                )
                x_G_b_pore, det_J_b_time_weight_pore = (
                    x_G_and_def_J_time_weight_2d_fuelcell_boundary(
                        cell_nodes_right_pore_y, x_max, x_G_line, weight_G_line
                    )
                )
                (
                    x_G_b_interface_electrode_electrolyte,
                    det_J_b_time_weight_interface_electrode_electrolyte,
                ) = x_G_b_and_det_J_b_time_weight_2d_fuelcell_boundary(
                    cell_nodes_interface_electrode_electrolyte_x,
                    cell_nodes_interface_electrode_electrolyte_y,
                    x_G_line,
                    weight_G_line,
                )
                (
                    x_G_b_interface_electrode_pore,
                    det_J_b_time_weight_interface_electrode_pore,
                ) = x_G_b_and_det_J_b_time_weight_2d_fuelcell_boundary(
                    cell_nodes_interface_electrode_pore_x,
                    cell_nodes_interface_electrode_pore_y,
                    x_G_line,
                    weight_G_line,
                )
                if delta_point_source == "False":
                    (
                        x_G_b_distributed_point_source_line,
                        det_J_b_time_weight_distributed_point_source_line,
                    ) = x_G_b_and_det_J_b_time_weight_2d_fuelcell_boundary(
                        np.asarray(cell_nodes_distributed_point_source_line_x),
                        np.asarray(cell_nodes_distributed_point_source_line_y),
                        x_G_line,
                        weight_G_line,
                    )

        if dimention == 3:
            x_G_electrolyte, det_J_time_weight_electrolyte = (
                x_G_and_def_J_time_weight_3d_fuelcell_domain(
                    cell_nodes_electrolyte_x,
                    cell_nodes_electrolyte_y,
                    cell_nodes_electrolyte_z,
                    x_G_cube,
                    weight_G_cube,
                )
            )
            x_G_electrode, det_J_time_weight_electrode = (
                x_G_and_def_J_time_weight_3d_fuelcell_domain(
                    cell_nodes_electrode_x,
                    cell_nodes_electrode_y,
                    cell_nodes_electrode_z,
                    x_G_cube,
                    weight_G_cube,
                )
            )
            x_G_pore, det_J_time_weight_pore = (
                x_G_and_def_J_time_weight_3d_fuelcell_domain(
                    cell_nodes_pore_x,
                    cell_nodes_pore_y,
                    cell_nodes_pore_z,
                    x_G_cube,
                    weight_G_cube,
                )
            )

            if single_grain == "True":
                x_G_b_electrolyte, det_J_b_time_weight_electrolyte = (
                    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
                        cell_nodes_left_electrolyte_x,
                        cell_nodes_left_electrolyte_z,
                        y_min_electrolyte,
                        x_G_rec,
                        weight_G_rec,
                    )
                )
                x_G_b_electrode, det_J_b_time_weight_electrode = (
                    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
                        cell_nodes_right_electrode_x,
                        cell_nodes_right_electrode_z,
                        y_max_electrode,
                        x_G_rec,
                        weight_G_rec,
                    )
                )
            else:
                # on left boundary of electrolyte
                x_G_b_electrolyte, det_J_b_time_weight_electrolyte = (
                    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
                        cell_nodes_left_electrolyte_x,
                        cell_nodes_left_electrolyte_z,
                        y_min,
                        x_G_rec,
                        weight_G_rec,
                    )
                )
                # on right boundary of electrode
                x_G_b_electrode, det_J_b_time_weight_electrode = (
                    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
                        cell_nodes_right_electrode_x,
                        cell_nodes_right_electrode_z,
                        y_max,
                        x_G_rec,
                        weight_G_rec,
                    )
                )
                # on right boundary of pore
                x_G_b_pore, det_J_b_time_weight_pore = (
                    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
                        cell_nodes_right_pore_x,
                        cell_nodes_right_pore_z,
                        y_max,
                        x_G_rec,
                        weight_G_rec,
                    )
                )
                # on electrolyte boundary and electrolyte/electrode interface
                (
                    x_G_b_interface_electrode_electrolyte,
                    det_J_b_time_weight_interface_electrode_electrolyte,
                ) = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(
                    cell_nodes_interface_electrode_electrolyte_x,
                    cell_nodes_interface_electrode_electrolyte_y,
                    cell_nodes_interface_electrode_electrolyte_z,
                    x_G_rec,
                    weight_G_rec,
                )
                # # on electrode boundary and electrolyte/electrode interface
                # x_G_b_interface_electrode_electrolyte_electrode, det_J_b_time_weight_interface_electrode_electrolyte_electrode = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_interface_electrode_electrolyte_electrode_x,cell_nodes_interface_electrode_electrolyte_electrode_y, cell_nodes_interface_electrode_electrolyte_electrode_z, x_G_rec, weight_G_rec)
                # on electrode boundary and pore/electrode interface
                (
                    x_G_b_interface_electrode_pore,
                    det_J_b_time_weight_interface_electrode_pore,
                ) = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(
                    cell_nodes_interface_electrode_pore_x,
                    cell_nodes_interface_electrode_pore_y,
                    cell_nodes_interface_electrode_pore_z,
                    x_G_rec,
                    weight_G_rec,
                )
                # # on pore boundary and pore/electrode interface
                # x_G_b_interface_electrode_pore_pore, det_J_b_time_weight_interface_electrode_pore_pore = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_interface_electrode_pore_pore_x,cell_nodes_interface_electrode_pore_pore_y, cell_nodes_interface_electrode_pore_pore_z, x_G_rec, weight_G_rec)
                if delta_point_source == "False":
                    (
                        x_G_b_distributed_point_source_surface,
                        det_J_b_time_weight_distributed_point_source_surface,
                    ) = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(
                        np.asarray(cell_nodes_distributed_point_source_surface_x),
                        np.asarray(cell_nodes_distributed_point_source_surface_y),
                        np.asarray(cell_nodes_distributed_point_source_surface_z),
                        x_G_rec,
                        weight_G_rec,
                    )

            x_G_b_line, det_J_b_time_weight_line = (
                x_G_and_det_J_line_3d_fuelcell_1d_boundary(
                    segments_source_coords, x_G_line, weight_G_line
                )
            )
            x_G_b_line = np.array(x_G_b_line)
            num_source_line_gauss_points = np.shape(x_G_b_line)[0]
            det_J_b_time_weight_line = np.array(det_J_b_time_weight_line)

            x_G_b_line_fixed, det_J_b_time_weight_line_fixed = (
                x_G_and_det_J_line_3d_fuelcell_1d_boundary(
                    segments_fixed_coods, x_G_line, weight_G_line
                )
            )
            x_G_b_line_fixed = np.array(x_G_b_line_fixed)
            num_fixed_line_gauss_points = np.shape(x_G_b_line_fixed)[0]
            det_J_b_time_weight_line_fixed = np.array(det_J_b_time_weight_line_fixed)

            gauss_rotation_axis_electrolyte = np.zeros((len(x_G_electrolyte), 3))
            gauss_rotation_axis_electrode = np.zeros((len(x_G_electrode), 3))
            gauss_rotation_axis_electrolyte[:, 0] = 1.0
            gauss_rotation_axis_electrode[:, 0] = 1.0

            gauss_rotation_axis = np.concatenate(
                (gauss_rotation_axis_electrolyte, gauss_rotation_axis_electrode), axis=0
            )

        num_gauss_points_in_domain_electrolyte = np.shape(x_G_electrolyte)[0]
        num_gauss_points_in_domain_electrode = np.shape(x_G_electrode)[0]
        num_gauss_points_in_domain_pore = np.shape(x_G_pore)[0]
        num_gauss_points_on_boundary_electrolyte = np.shape(x_G_b_electrolyte)[0]
        num_gauss_points_on_boundary_electrode = np.shape(x_G_b_electrode)[0]
        num_gauss_points_on_boundary_pore = np.shape(x_G_b_pore)[0]
        num_gauss_points_on_electrolyte_electrode_interface = np.shape(
            x_G_b_interface_electrode_electrolyte
        )[0]
        num_gauss_points_on_electrode_pore_interface = np.shape(
            x_G_b_interface_electrode_pore
        )[0]

        gauss_angle_electrolyte = angle * np.ones(
            num_gauss_points_in_domain_electrolyte
        )
        gauss_angle_electrode = angle * np.ones(num_gauss_points_in_domain_electrode)
        gauss_angle_pore = angle * np.ones(num_gauss_points_in_domain_pore)
        gauss_angle_b_electrolyte = angle * np.ones(
            num_gauss_points_on_boundary_electrolyte
        )
        gauss_angle_b_electrode = angle * np.ones(
            num_gauss_points_on_boundary_electrode
        )
        gauss_angle_b_pore = angle * np.ones(num_gauss_points_on_boundary_pore)
        gauss_angle_electrolyte_electrode_interface = angle * np.ones(
            num_gauss_points_on_electrolyte_electrode_interface
        )
        gauss_angle_electrode_pore_interface = angle * np.ones(
            num_gauss_points_on_electrode_pore_interface
        )

        Gauss_grain_id_electrolyte = 1 * np.ones(num_gauss_points_in_domain_electrolyte)
        Gauss_grain_id_electrode = 1 * np.ones(num_gauss_points_in_domain_electrode)
        Gauss_grain_id_pore = 1 * np.ones(num_gauss_points_in_domain_pore)
        Gauss_b_grain_id_electrolyte = 1 * np.ones(
            num_gauss_points_on_boundary_electrolyte
        )
        Gauss_b_grain_id_electrode = 1 * np.ones(num_gauss_points_on_boundary_electrode)
        Gauss_b_grain_id_pore = 1 * np.ones(num_gauss_points_on_boundary_pore)
        Gauss_b_grain_id_electrolyte_electrode_interace = 1 * np.ones(
            num_gauss_points_on_electrolyte_electrode_interface
        )
        Gauss_b_grain_id_electrode_pore_interace = 1 * np.ones(
            num_gauss_points_on_electrode_pore_interface
        )

        (
            x_G_electrolyte,
            x_G_b_electrolyte,
            x_G_electrode,
            x_G_b_electrode,
            x_G_pore,
            x_G_b_pore,
            x_G_b_interface_electrode_electrolyte,
            x_G_b_interface_electrode_pore,
        ) = [
            np.array(lst)
            for lst in [
                x_G_electrolyte,
                x_G_b_electrolyte,
                x_G_electrode,
                x_G_b_electrode,
                x_G_pore,
                x_G_b_pore,
                x_G_b_interface_electrode_electrolyte,
                x_G_b_interface_electrode_pore,
            ]
        ]

        (
            gauss_angle_electrolyte,
            gauss_angle_b_electrolyte,
            gauss_angle_electrode,
            gauss_angle_b_electrode,
            gauss_angle_pore,
            gauss_angle_b_pore,
            gauss_angle_electrolyte_electrode_interface,
            gauss_angle_electrode_pore_interface,
        ) = [
            np.array(lst)
            for lst in [
                gauss_angle_electrolyte,
                gauss_angle_b_electrolyte,
                gauss_angle_electrode,
                gauss_angle_b_electrode,
                gauss_angle_pore,
                gauss_angle_b_pore,
                gauss_angle_electrolyte_electrode_interface,
                gauss_angle_electrode_pore_interface,
            ]
        ]
        (
            Gauss_grain_id_electrolyte,
            Gauss_b_grain_id_electrolyte,
            Gauss_grain_id_electrode,
            Gauss_b_grain_id_electrode,
            Gauss_grain_id_pore,
            Gauss_b_grain_id_pore,
            Gauss_b_grain_id_electrolyte_electrode_interace,
            Gauss_b_grain_id_electrode_pore_interace,
        ) = [
            np.array(lst)
            for lst in [
                Gauss_grain_id_electrolyte,
                Gauss_b_grain_id_electrolyte,
                Gauss_grain_id_electrode,
                Gauss_b_grain_id_electrode,
                Gauss_grain_id_pore,
                Gauss_b_grain_id_pore,
                Gauss_b_grain_id_electrolyte_electrode_interace,
                Gauss_b_grain_id_electrode_pore_interace,
            ]
        ]

        # all gauss points in domain, used for mechanical simulation
        x_G_mechanical = np.concatenate((x_G_electrolyte, x_G_electrode), axis=0)
        det_J_time_weight_mechanical = np.concatenate(
            (
                np.asarray(det_J_time_weight_electrolyte),
                np.asarray(det_J_time_weight_electrode),
            ),
            axis=0,
        )
        Gauss_grain_id_mechanical = np.concatenate(
            (Gauss_grain_id_electrolyte, Gauss_grain_id_electrode), axis=0
        )
        Gauss_angle_mechanical = np.concatenate(
            (gauss_angle_electrolyte, gauss_angle_electrode), axis=0
        )
        num_gauss_points_in_domain_mechanical = np.shape(x_G_mechanical)[0]

        print(
            "number of Gauss points in electrolyte domain: "
            + str(num_gauss_points_in_domain_electrolyte)
        )
        print(
            "number of Gauss points on electrolyte boundaries: "
            + str(num_gauss_points_on_boundary_electrolyte)
        )
        print(
            "number of Gauss points in electrode domain: "
            + str(num_gauss_points_in_domain_electrode)
        )
        print(
            "number of Gauss points on electrode boundaries: "
            + str(num_gauss_points_on_boundary_electrode)
        )

def_nodes_gauss_points_time = time.time()
print(
    "time to define nodes and Gauss points = "
    + "%s seconds" % (def_nodes_gauss_points_time - def_para_time)
)

####################################################
# Compute shape function and its gradient in domain
#####################################################
print("Compute shape function and its gradient in domain")

c = 2  # support size

if dimention == 2:
    HT0 = np.array([1, 0, 0], dtype=np.float64)  # transpose of the basis vector H
    HT1 = np.array(
        [0, -1, 0], dtype=np.float64
    )  # for computation of gradient of shape function, d/dx
    HT2 = np.array(
        [0, 0, -1], dtype=np.float64
    )  # for computation of gradient of shape function, d/dy
if dimention == 3:
    HT0 = np.array([1, 0, 0, 0], dtype=np.float64)  # transpose of the basis vector H
    HT1 = np.array(
        [0, -1, 0, 0], dtype=np.float64
    )  # for computation of gradient of shape function, d/dx
    HT2 = np.array(
        [0, 0, -1, 0], dtype=np.float64
    )  # for computation of gradient of shape function, d/dy
    HT3 = np.array(
        [0, 0, 0, -1], dtype=np.float64
    )  # for computation of gradient of shape function, d/dy

if studied_physics == "battery" and single_grain == "True":
    a = (
        c * (x_max - x_min) / n_intervals * np.ones(num_nodes)
    )  # compact support size, shape: (num_nodes,)
if studied_physics == "fuel cell" and single_grain == "True":
    a_electrode = (
        c
        * (x_max_electrode - x_min_electrode)
        / n_intervals
        * np.ones(num_nodes_electrode)
    )  # compact support size, shape: (num_nodes,)
    a_electrolyte = a_electrode
    a_pore = a_electrode
    a_mechanical = (
        c
        * (x_max_electrode - x_min_electrode)
        / n_intervals
        * np.ones(num_nodes_mechanical)
    )
if studied_physics == "fuel cell" and single_grain == "False":
    a_electrolyte = (
        c * (x_max - x_min) / num_pixels_xyz[0] * np.ones(num_nodes_electrolyte)
    )  # compact support size, shape: (num_nodes,)
    a_electrode = c * (x_max - x_min) / num_pixels_xyz[0] * np.ones(num_nodes_electrode)
    a_pore = c * (x_max - x_min) / num_pixels_xyz[0] * np.ones(num_nodes_pore)
    a_mechanical = (
        c * (x_max - x_min) / num_pixels_xyz[0] * np.ones(num_nodes_mechanical)
    )

if studied_physics == "battery" and single_grain == "False":

    h = np.zeros(num_nodes)
    for i in range(num_nodes):
        dist = (
            (x_nodes[i, 0] - x_nodes[:, 0]) ** 2 + (x_nodes[i, 1] - x_nodes[:, 1]) ** 2
        ) ** 0.5

        index_four_smallest = sorted(range(len(dist)), key=lambda sub: dist[sub])[
            :5
        ]  # get the index of the four smallest index, the first one is always zero, so 5 here

        h[i] = dist[index_four_smallest][
            dist[index_four_smallest].tolist().index(max(dist[index_four_smallest]))
        ]

    a = c * h  # shape: (num_nodes,)

if studied_physics == "battery":
    M = np.array([np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain)])
    M_P_x = np.array(
        [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain)]
    )  # partial M partial x
    M_P_y = np.array(
        [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain)]
    )  # partial M partial y

    (
        phi_nonzero_index_row,
        phi_nonzero_index_column,
        phi_nonzerovalue_data,
        phi_P_x_nonzerovalue_data,
        phi_P_y_nonzerovalue_data,
        phi_P_z_nonzerovalue_data,
        M,
        M_P_x,
        M_P_y,
        M_P_z,
    ) = compute_phi_M(
        x_G,
        Gauss_grain_id,
        x_nodes,
        nodes_grain_id,
        a,
        M,
        M_P_x,
        M_P_y,
        num_interface_segments,
        interface_nodes,
        BxByCxCy,
        IM_RKPM,
        single_grain,
    )

    num_non_zero_phi_a = np.shape(np.array(phi_nonzero_index_row))[0]

    (
        shape_func_value,
        shape_func_times_det_J_time_weight_value,
        grad_shape_func_x_value,
        grad_shape_func_y_value,
        grad_shape_func_z_value,
        grad_shape_func_x_times_det_J_time_weight_value,
        grad_shape_func_y_times_det_J_time_weight_value,
        grad_shape_func_z_times_det_J_time_weight_value,
    ) = shape_grad_shape_func(
        x_G,
        x_nodes,
        num_non_zero_phi_a,
        HT0,
        M,
        M_P_x,
        M_P_y,
        differential_method,
        HT1,
        HT2,
        phi_nonzerovalue_data,
        phi_P_x_nonzerovalue_data,
        phi_P_y_nonzerovalue_data,
        phi_nonzero_index_row,
        phi_nonzero_index_column,
        det_J_time_weight,
        IM_RKPM,
    )

    # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
    shape_func = csc_matrix(
        (
            np.array(shape_func_value),
            (np.array(phi_nonzero_index_row), np.array(phi_nonzero_index_column)),
        ),
        shape=(num_gauss_points_in_domain, num_nodes),
    )
    shape_func_times_det_J_time_weight = csc_matrix(
        (
            np.array(shape_func_times_det_J_time_weight_value),
            (np.array(phi_nonzero_index_row), np.array(phi_nonzero_index_column)),
        ),
        shape=(num_gauss_points_in_domain, num_nodes),
    )
    grad_shape_func_x = csc_matrix(
        (
            np.array(grad_shape_func_x_value),
            (np.array(phi_nonzero_index_row), np.array(phi_nonzero_index_column)),
        ),
        shape=(num_gauss_points_in_domain, num_nodes),
    )
    grad_shape_func_y = csc_matrix(
        (
            np.array(grad_shape_func_y_value),
            (np.array(phi_nonzero_index_row), np.array(phi_nonzero_index_column)),
        ),
        shape=(num_gauss_points_in_domain, num_nodes),
    )
    grad_shape_func_x_times_det_J_time_weight = csc_matrix(
        (
            np.array(grad_shape_func_x_times_det_J_time_weight_value),
            (np.array(phi_nonzero_index_row), np.array(phi_nonzero_index_column)),
        ),
        shape=(num_gauss_points_in_domain, num_nodes),
    )
    grad_shape_func_y_times_det_J_time_weight = csc_matrix(
        (
            np.array(grad_shape_func_y_times_det_J_time_weight_value),
            (np.array(phi_nonzero_index_row), np.array(phi_nonzero_index_column)),
        ),
        shape=(num_gauss_points_in_domain, num_nodes),
    )
else:
    if dimention == 2:
        M_electrolyte = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )
        M_P_x_electrolyte = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )  # partial M partial x
        M_P_y_electrolyte = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )  # partial M partial y

        M_electrode = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_electrode)]
        )
        M_P_x_electrode = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_electrode)]
        )  # partial M partial x
        M_P_y_electrode = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_electrode)]
        )  # partial M partial y

        M_pore = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_pore)]
        )
        M_P_x_pore = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_pore)]
        )  # partial M partial x
        M_P_y_pore = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_pore)]
        )  # partial M partial y

        M_mechanical = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )
        M_P_x_mechanical = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )
        M_P_y_mechanical = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )

        (
            phi_nonzero_index_row_electrolyte,
            phi_nonzero_index_column_electrolyte,
            phi_nonzerovalue_data_electrolyte,
            phi_P_x_nonzerovalue_data_electrolyte,
            phi_P_y_nonzerovalue_data_electrolyte,
            phi_P_z_nonzerovalue_data_electrolyte,
            M_electrolyte,
            M_P_x_electrolyte,
            M_P_y_electrolyte,
            M_P_z_electrolyte,
        ) = compute_phi_M(
            x_G_electrolyte,
            Gauss_grain_id_electrolyte,
            x_nodes_electrolyte,
            nodes_grain_id_electrolyte,
            a_electrolyte,
            M_electrolyte,
            M_P_x_electrolyte,
            M_P_y_electrolyte,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        (
            phi_nonzero_index_row_electrode,
            phi_nonzero_index_column_electrode,
            phi_nonzerovalue_data_electrode,
            phi_P_x_nonzerovalue_data_electrode,
            phi_P_y_nonzerovalue_data_electrode,
            phi_P_z_nonzerovalue_data_electrode,
            M_electrode,
            M_P_x_electrode,
            M_P_y_electrode,
            M_P_z_electrode,
        ) = compute_phi_M(
            x_G_electrode,
            Gauss_grain_id_electrode,
            x_nodes_electrode,
            nodes_grain_id_electrode,
            a_electrode,
            M_electrode,
            M_P_x_electrode,
            M_P_y_electrode,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        (
            phi_nonzero_index_row_pore,
            phi_nonzero_index_column_pore,
            phi_nonzerovalue_data_pore,
            phi_P_x_nonzerovalue_data_pore,
            phi_P_y_nonzerovalue_data_pore,
            phi_P_z_nonzerovalue_data_pore,
            M_pore,
            M_P_x_pore,
            M_P_y_pore,
            M_P_z_pore,
        ) = compute_phi_M(
            x_G_pore,
            Gauss_grain_id_pore,
            x_nodes_pore,
            nodes_grain_id_pore,
            a_pore,
            M_pore,
            M_P_x_pore,
            M_P_y_pore,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        (
            phi_nonzero_index_row_mechanical,
            phi_nonzero_index_column_mechanical,
            phi_nonzerovalue_data_mechanical,
            phi_P_x_nonzerovalue_data_mechanical,
            phi_P_y_nonzerovalue_data_mechanical,
            phi_P_z_nonzerovalue_data_mechanical,
            M_mechanical,
            M_P_x_mechanical,
            M_P_y_mechanical,
            M_P_z_mechanical,
        ) = compute_phi_M(
            x_G_mechanical,
            Gauss_grain_id_mechanical,
            x_nodes_mechanical,
            nodes_grain_id_mechanical,
            a_mechanical,
            M_mechanical,
            M_P_x_mechanical,
            M_P_y_mechanical,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

    if dimention == 3:
        M_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )
        M_P_x_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )  # partial M partial x
        M_P_y_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )  # partial M partial y
        M_P_z_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )  # partial M partial y

        M_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrode)]
        )
        M_P_x_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrode)]
        )  # partial M partial x
        M_P_y_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrode)]
        )  # partial M partial y
        M_P_z_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_electrolyte)]
        )  # partial M partial y

        M_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_pore)]
        )
        M_P_x_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_pore)]
        )  # partial M partial x
        M_P_y_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_pore)]
        )  # partial M partial y
        M_P_z_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_pore)]
        )  # partial M partial y

        M_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )
        M_P_x_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )  # partial M partial x
        M_P_y_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )  # partial M partial y
        M_P_z_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_in_domain_mechanical)]
        )  # partial M partial y

        # phi_nonzero_index_row_electrolyte, phi_nonzero_index_column_electrolyte, phi_nonzerovalue_data_electrolyte, phi_P_x_nonzerovalue_data_electrolyte, phi_P_y_nonzerovalue_data_electrolyte,phi_P_z_nonzerovalue_data_electrolyte, M_electrolyte, M_P_x_electrolyte, M_P_y_electrolyte,M_P_z_electrolyte = compute_phi_M(x_G_electrolyte, Gauss_grain_id_electrolyte, x_nodes_electrolyte,nodes_grain_id_electrolyte, a_electrolyte, M_electrolyte, M_P_x_electrolyte, M_P_y_electrolyte, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrolyte)
        # phi_nonzero_index_row_electrode, phi_nonzero_index_column_electrode, phi_nonzerovalue_data_electrode, phi_P_x_nonzerovalue_data_electrode, phi_P_y_nonzerovalue_data_electrode,phi_P_z_nonzerovalue_data_electrode, M_electrode, M_P_x_electrode, M_P_y_electrode,M_P_z_electrode = compute_phi_M(x_G_electrode, Gauss_grain_id_electrode, x_nodes_electrode,nodes_grain_id_electrode, a_electrode, M_electrode, M_P_x_electrode, M_P_y_electrode, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrode)
        # phi_nonzero_index_row_pore, phi_nonzero_index_column_pore, phi_nonzerovalue_data_pore, phi_P_x_nonzerovalue_data_pore, phi_P_y_nonzerovalue_data_pore,phi_P_z_nonzerovalue_data_pore, M_pore, M_P_x_pore, M_P_y_pore,M_P_z_pore = compute_phi_M(x_G_pore, Gauss_grain_id_pore, x_nodes_pore,nodes_grain_id_pore, a_pore, M_pore, M_P_x_pore, M_P_y_pore, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_pore)
        # phi_nonzero_index_row_mechanical, phi_nonzero_index_column_mechanical, phi_nonzerovalue_data_mechanical, phi_P_x_nonzerovalue_data_mechanical, phi_P_y_nonzerovalue_data_mechanical,phi_P_z_nonzerovalue_data_mechanical, M_mechanical, M_P_x_mechanical, M_P_y_mechanical,M_P_z_mechanical = compute_phi_M(x_G_mechanical, Gauss_grain_id_mechanical, x_nodes_mechanical,nodes_grain_id_mechanical, a_mechanical, M_mechanical, M_P_x_mechanical, M_P_y_mechanical, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_mechanical)

    # np.savetxt('phi_nonzero_index_row_electrolyte.txt', phi_nonzero_index_row_electrolyte)
    # np.savetxt('phi_nonzero_index_column_electrolyte.txt', phi_nonzero_index_column_electrolyte)
    phi_nonzero_index_row_electrolyte = np.loadtxt(
        "phi_nonzero_index_row_electrolyte.txt"
    )
    phi_nonzero_index_column_electrolyte = np.loadtxt(
        "phi_nonzero_index_column_electrolyte.txt"
    )

    # np.savetxt('phi_nonzero_index_row_electrode.txt', phi_nonzero_index_row_electrode)
    # np.savetxt('phi_nonzero_index_column_electrode.txt', phi_nonzero_index_column_electrode)
    phi_nonzero_index_row_electrode = np.loadtxt("phi_nonzero_index_row_electrode.txt")
    phi_nonzero_index_column_electrode = np.loadtxt(
        "phi_nonzero_index_column_electrode.txt"
    )

    # np.savetxt('phi_nonzero_index_row_pore.txt', phi_nonzero_index_row_pore)
    # np.savetxt('phi_nonzero_index_column_pore.txt', phi_nonzero_index_column_pore)
    phi_nonzero_index_row_pore = np.loadtxt("phi_nonzero_index_row_pore.txt")
    phi_nonzero_index_column_pore = np.loadtxt("phi_nonzero_index_column_pore.txt")

    # np.savetxt('phi_nonzero_index_row_mechanical.txt', phi_nonzero_index_row_mechanical)
    # np.savetxt('phi_nonzero_index_column_mechanical.txt', phi_nonzero_index_column_mechanical)
    phi_nonzero_index_row_mechanical = np.loadtxt(
        "phi_nonzero_index_row_mechanical.txt"
    )
    phi_nonzero_index_column_mechanical = np.loadtxt(
        "phi_nonzero_index_column_mechanical.txt"
    )

    num_non_zero_phi_a_electrolyte = np.shape(
        np.array(phi_nonzero_index_row_electrolyte)
    )[0]
    num_non_zero_phi_a_electrode = np.shape(np.array(phi_nonzero_index_row_electrode))[
        0
    ]
    num_non_zero_phi_a_pore = np.shape(np.array(phi_nonzero_index_row_pore))[0]
    num_non_zero_phi_a_mechanical = np.shape(
        np.array(phi_nonzero_index_row_mechanical)
    )[0]

    if dimention == 2:
        (
            shape_func_value_electrolyte,
            shape_func_times_det_J_time_weight_value_electrolyte,
            grad_shape_func_x_value_electrolyte,
            grad_shape_func_y_value_electrolyte,
            grad_shape_func_z_value_electrolyte,
            grad_shape_func_x_times_det_J_time_weight_value_electrolyte,
            grad_shape_func_y_times_det_J_time_weight_value_electrolyte,
            grad_shape_func_z_times_det_J_time_weight_value_electrolyte,
        ) = shape_grad_shape_func(
            x_G_electrolyte,
            x_nodes_electrolyte,
            num_non_zero_phi_a_electrolyte,
            HT0,
            M_electrolyte,
            M_P_x_electrolyte,
            M_P_y_electrolyte,
            differential_method,
            HT1,
            HT2,
            phi_nonzerovalue_data_electrolyte,
            phi_P_x_nonzerovalue_data_electrolyte,
            phi_P_y_nonzerovalue_data_electrolyte,
            phi_nonzero_index_row_electrolyte,
            phi_nonzero_index_column_electrolyte,
            det_J_time_weight_electrolyte,
            IM_RKPM,
        )
        (
            shape_func_value_electrode,
            shape_func_times_det_J_time_weight_value_electrode,
            grad_shape_func_x_value_electrode,
            grad_shape_func_y_value_electrode,
            grad_shape_func_z_value_electrode,
            grad_shape_func_x_times_det_J_time_weight_value_electrode,
            grad_shape_func_y_times_det_J_time_weight_value_electrode,
            grad_shape_func_z_times_det_J_time_weight_value_electrode,
        ) = shape_grad_shape_func(
            x_G_electrode,
            x_nodes_electrode,
            num_non_zero_phi_a_electrode,
            HT0,
            M_electrode,
            M_P_x_electrode,
            M_P_y_electrode,
            differential_method,
            HT1,
            HT2,
            phi_nonzerovalue_data_electrode,
            phi_P_x_nonzerovalue_data_electrode,
            phi_P_y_nonzerovalue_data_electrode,
            phi_nonzero_index_row_electrode,
            phi_nonzero_index_column_electrode,
            det_J_time_weight_electrode,
            IM_RKPM,
        )
        (
            shape_func_value_pore,
            shape_func_times_det_J_time_weight_value_pore,
            grad_shape_func_x_value_pore,
            grad_shape_func_y_value_pore,
            grad_shape_func_z_value_pore,
            grad_shape_func_x_times_det_J_time_weight_value_pore,
            grad_shape_func_y_times_det_J_time_weight_value_pore,
            grad_shape_func_z_times_det_J_time_weight_value_pore,
        ) = shape_grad_shape_func(
            x_G_pore,
            x_nodes_pore,
            num_non_zero_phi_a_pore,
            HT0,
            M_pore,
            M_P_x_pore,
            M_P_y_pore,
            differential_method,
            HT1,
            HT2,
            phi_nonzerovalue_data_pore,
            phi_P_x_nonzerovalue_data_pore,
            phi_P_y_nonzerovalue_data_pore,
            phi_nonzero_index_row_pore,
            phi_nonzero_index_column_pore,
            det_J_time_weight_pore,
            IM_RKPM,
        )
        (
            shape_func_value_mechanical,
            shape_func_times_det_J_time_weight_value_mechanical,
            grad_shape_func_x_value_mechanical,
            grad_shape_func_y_value_mechanical,
            grad_shape_func_z_value_mechanical,
            grad_shape_func_x_times_det_J_time_weight_value_mechanical,
            grad_shape_func_y_times_det_J_time_weight_value_mechanical,
            grad_shape_func_z_times_det_J_time_weight_value_mechanical,
        ) = shape_grad_shape_func(
            x_G_mechanical,
            x_nodes_mechanical,
            num_non_zero_phi_a_mechanical,
            HT0,
            M_mechanical,
            M_P_x_mechanical,
            M_P_y_mechanical,
            differential_method,
            HT1,
            HT2,
            phi_nonzerovalue_data_mechanical,
            phi_P_x_nonzerovalue_data_mechanical,
            phi_P_y_nonzerovalue_data_mechanical,
            phi_nonzero_index_row_mechanical,
            phi_nonzero_index_column_mechanical,
            det_J_time_weight_mechanical,
            IM_RKPM,
        )

    if dimention == 3:
        print("yes")
        # shape_func_value_electrolyte, shape_func_times_det_J_time_weight_value_electrolyte, grad_shape_func_x_value_electrolyte, grad_shape_func_y_value_electrolyte,grad_shape_func_z_value_electrolyte, grad_shape_func_x_times_det_J_time_weight_value_electrolyte, grad_shape_func_y_times_det_J_time_weight_value_electrolyte,grad_shape_func_z_times_det_J_time_weight_value_electrolyte = shape_grad_shape_func(x_G_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_electrolyte,HT0, M_electrolyte, M_P_x_electrolyte, M_P_y_electrolyte, differential_method, HT1, HT2, phi_nonzerovalue_data_electrolyte,phi_P_x_nonzerovalue_data_electrolyte,phi_P_y_nonzerovalue_data_electrolyte, phi_nonzero_index_row_electrolyte, phi_nonzero_index_column_electrolyte, det_J_time_weight_electrolyte, IM_RKPM, M_P_z_electrolyte, HT3, phi_P_z_nonzerovalue_data_electrolyte)
        # shape_func_value_electrode, shape_func_times_det_J_time_weight_value_electrode, grad_shape_func_x_value_electrode,grad_shape_func_y_value_electrode, grad_shape_func_z_value_electrode, grad_shape_func_x_times_det_J_time_weight_value_electrode, grad_shape_func_y_times_det_J_time_weight_value_electrode, grad_shape_func_z_times_det_J_time_weight_value_electrode = shape_grad_shape_func(x_G_electrode,x_nodes_electrode, num_non_zero_phi_a_electrode,HT0, M_electrode, M_P_x_electrode, M_P_y_electrode, differential_method, HT1, HT2, phi_nonzerovalue_data_electrode,phi_P_x_nonzerovalue_data_electrode,phi_P_y_nonzerovalue_data_electrode, phi_nonzero_index_row_electrode, phi_nonzero_index_column_electrode, det_J_time_weight_electrode, IM_RKPM, M_P_z_electrode, HT3, phi_P_z_nonzerovalue_data_electrode)
        # shape_func_value_pore, shape_func_times_det_J_time_weight_value_pore, grad_shape_func_x_value_pore,grad_shape_func_y_value_pore, grad_shape_func_z_value_pore, grad_shape_func_x_times_det_J_time_weight_value_pore, grad_shape_func_y_times_det_J_time_weight_value_pore, grad_shape_func_z_times_det_J_time_weight_value_pore = shape_grad_shape_func(x_G_pore,x_nodes_pore, num_non_zero_phi_a_pore,HT0, M_pore, M_P_x_pore, M_P_y_pore, differential_method, HT1, HT2, phi_nonzerovalue_data_pore,phi_P_x_nonzerovalue_data_pore,phi_P_y_nonzerovalue_data_pore, phi_nonzero_index_row_pore, phi_nonzero_index_column_pore, det_J_time_weight_pore, IM_RKPM, M_P_z_pore, HT3, phi_P_z_nonzerovalue_data_pore)
        # shape_func_value_mechanical, shape_func_times_det_J_time_weight_value_mechanical, grad_shape_func_x_value_mechanical,grad_shape_func_y_value_mechanical, grad_shape_func_z_value_mechanical, grad_shape_func_x_times_det_J_time_weight_value_mechanical, grad_shape_func_y_times_det_J_time_weight_value_mechanical, grad_shape_func_z_times_det_J_time_weight_value_mechanical = shape_grad_shape_func(x_G_mechanical,x_nodes_mechanical, num_non_zero_phi_a_mechanical,HT0, M_mechanical, M_P_x_mechanical, M_P_y_mechanical, differential_method, HT1, HT2, phi_nonzerovalue_data_mechanical,phi_P_x_nonzerovalue_data_mechanical,phi_P_y_nonzerovalue_data_mechanical, phi_nonzero_index_row_mechanical, phi_nonzero_index_column_mechanical, det_J_time_weight_mechanical, IM_RKPM, M_P_z_mechanical, HT3, phi_P_z_nonzerovalue_data_mechanical)

    # np.savetxt('shape_func_value_electrolyte.txt', np.asarray(shape_func_value_electrolyte))
    # np.savetxt('shape_func_times_det_J_time_weight_value_electrolyte.txt', np.asarray(shape_func_times_det_J_time_weight_value_electrolyte))
    # np.savetxt('grad_shape_func_x_value_electrolyte.txt', np.asarray(grad_shape_func_x_value_electrolyte))
    # np.savetxt('grad_shape_func_y_value_electrolyte.txt', np.asarray(grad_shape_func_y_value_electrolyte))
    # np.savetxt('grad_shape_func_z_value_electrolyte.txt', np.asarray(grad_shape_func_z_value_electrolyte))
    # np.savetxt('grad_shape_func_x_times_det_J_time_weight_value_electrolyte.txt', np.asarray(grad_shape_func_x_times_det_J_time_weight_value_electrolyte))
    # np.savetxt('grad_shape_func_y_times_det_J_time_weight_value_electrolyte.txt', np.asarray(grad_shape_func_y_times_det_J_time_weight_value_electrolyte))
    # np.savetxt('grad_shape_func_z_times_det_J_time_weight_value_electrolyte.txt', np.asarray(grad_shape_func_z_times_det_J_time_weight_value_electrolyte))
    shape_func_value_electrolyte = np.loadtxt("shape_func_value_electrolyte.txt")
    shape_func_times_det_J_time_weight_value_electrolyte = np.loadtxt(
        "shape_func_times_det_J_time_weight_value_electrolyte.txt"
    )
    grad_shape_func_x_value_electrolyte = np.loadtxt(
        "grad_shape_func_x_value_electrolyte.txt"
    )
    grad_shape_func_y_value_electrolyte = np.loadtxt(
        "grad_shape_func_y_value_electrolyte.txt"
    )
    grad_shape_func_z_value_electrolyte = np.loadtxt(
        "grad_shape_func_z_value_electrolyte.txt"
    )
    grad_shape_func_x_times_det_J_time_weight_value_electrolyte = np.loadtxt(
        "grad_shape_func_x_times_det_J_time_weight_value_electrolyte.txt"
    )
    grad_shape_func_y_times_det_J_time_weight_value_electrolyte = np.loadtxt(
        "grad_shape_func_y_times_det_J_time_weight_value_electrolyte.txt"
    )
    grad_shape_func_z_times_det_J_time_weight_value_electrolyte = np.loadtxt(
        "grad_shape_func_z_times_det_J_time_weight_value_electrolyte.txt"
    )

    # np.savetxt('shape_func_value_electrode.txt', np.asarray(shape_func_value_electrode))
    # np.savetxt('shape_func_times_det_J_time_weight_value_electrode.txt', np.asarray(shape_func_times_det_J_time_weight_value_electrode))
    # np.savetxt('grad_shape_func_x_value_electrode.txt', np.asarray(grad_shape_func_x_value_electrode))
    # np.savetxt('grad_shape_func_y_value_electrode.txt', np.asarray(grad_shape_func_y_value_electrode))
    # np.savetxt('grad_shape_func_z_value_electrode.txt', np.asarray(grad_shape_func_z_value_electrode))
    # np.savetxt('grad_shape_func_x_times_det_J_time_weight_value_electrode.txt.txt', np.asarray(grad_shape_func_x_times_det_J_time_weight_value_electrode))
    # np.savetxt('grad_shape_func_y_times_det_J_time_weight_value_electrode.txt.txt', np.asarray(grad_shape_func_y_times_det_J_time_weight_value_electrode))
    # np.savetxt('grad_shape_func_z_times_det_J_time_weight_value_electrode.txt.txt', np.asarray(grad_shape_func_z_times_det_J_time_weight_value_electrode))
    shape_func_value_electrode = np.loadtxt("shape_func_value_electrode.txt")
    shape_func_times_det_J_time_weight_value_electrode = np.loadtxt(
        "shape_func_times_det_J_time_weight_value_electrode.txt"
    )
    grad_shape_func_x_value_electrode = np.loadtxt(
        "grad_shape_func_x_value_electrode.txt"
    )
    grad_shape_func_y_value_electrode = np.loadtxt(
        "grad_shape_func_y_value_electrode.txt"
    )
    grad_shape_func_x_times_det_J_time_weight_value_electrode = np.loadtxt(
        "grad_shape_func_x_times_det_J_time_weight_value_electrode.txt.txt"
    )
    grad_shape_func_y_times_det_J_time_weight_value_electrode = np.loadtxt(
        "grad_shape_func_y_times_det_J_time_weight_value_electrode.txt.txt"
    )
    grad_shape_func_z_value_electrode = np.loadtxt(
        "grad_shape_func_z_value_electrode.txt"
    )
    grad_shape_func_z_times_det_J_time_weight_value_electrode = np.loadtxt(
        "grad_shape_func_z_times_det_J_time_weight_value_electrode.txt.txt"
    )

    # np.savetxt('shape_func_value_pore.txt', np.asarray(shape_func_value_pore))
    # np.savetxt('shape_func_times_det_J_time_weight_value_pore.txt', np.asarray(shape_func_times_det_J_time_weight_value_pore))
    # np.savetxt('grad_shape_func_x_value_pore.txt', np.asarray(grad_shape_func_x_value_pore))
    # np.savetxt('grad_shape_func_y_value_pore.txt', np.asarray(grad_shape_func_y_value_pore))
    # np.savetxt('grad_shape_func_z_value_pore.txt', np.asarray(grad_shape_func_z_value_pore))
    # np.savetxt('grad_shape_func_x_times_det_J_time_weight_value_pore.txt.txt', np.asarray(grad_shape_func_x_times_det_J_time_weight_value_pore))
    # np.savetxt('grad_shape_func_y_times_det_J_time_weight_value_pore.txt.txt', np.asarray(grad_shape_func_y_times_det_J_time_weight_value_pore))
    # np.savetxt('grad_shape_func_z_times_det_J_time_weight_value_pore.txt.txt', np.asarray(grad_shape_func_z_times_det_J_time_weight_value_pore))
    shape_func_value_pore = np.loadtxt("shape_func_value_pore.txt")
    shape_func_times_det_J_time_weight_value_pore = np.loadtxt(
        "shape_func_times_det_J_time_weight_value_pore.txt"
    )
    grad_shape_func_x_value_pore = np.loadtxt("grad_shape_func_x_value_pore.txt")
    grad_shape_func_y_value_pore = np.loadtxt("grad_shape_func_y_value_pore.txt")
    grad_shape_func_x_times_det_J_time_weight_value_pore = np.loadtxt(
        "grad_shape_func_x_times_det_J_time_weight_value_pore.txt.txt"
    )
    grad_shape_func_y_times_det_J_time_weight_value_pore = np.loadtxt(
        "grad_shape_func_y_times_det_J_time_weight_value_pore.txt.txt"
    )
    grad_shape_func_z_value_pore = np.loadtxt("grad_shape_func_z_value_pore.txt")
    grad_shape_func_z_times_det_J_time_weight_value_pore = np.loadtxt(
        "grad_shape_func_z_times_det_J_time_weight_value_pore.txt.txt"
    )

    # np.savetxt('shape_func_value_mechanical.txt', np.asarray(shape_func_value_mechanical))
    # np.savetxt('shape_func_times_det_J_time_weight_value_mechanical.txt', np.asarray(shape_func_times_det_J_time_weight_value_mechanical))
    # np.savetxt('grad_shape_func_x_value_mechanical.txt', np.asarray(grad_shape_func_x_value_mechanical))
    # np.savetxt('grad_shape_func_y_value_mechanical.txt', np.asarray(grad_shape_func_y_value_mechanical))
    # np.savetxt('grad_shape_func_z_value_mechanical.txt', np.asarray(grad_shape_func_z_value_mechanical))
    # np.savetxt('grad_shape_func_x_times_det_J_time_weight_value_mechanical.txt.txt', np.asarray(grad_shape_func_x_times_det_J_time_weight_value_mechanical))
    # np.savetxt('grad_shape_func_y_times_det_J_time_weight_value_mechanical.txt.txt', np.asarray(grad_shape_func_y_times_det_J_time_weight_value_mechanical))
    # np.savetxt('grad_shape_func_z_times_det_J_time_weight_value_mechanical.txt.txt', np.asarray(grad_shape_func_z_times_det_J_time_weight_value_mechanical))
    shape_func_value_mechanical = np.loadtxt("shape_func_value_mechanical.txt")
    shape_func_times_det_J_time_weight_value_mechanical = np.loadtxt(
        "shape_func_times_det_J_time_weight_value_mechanical.txt"
    )
    grad_shape_func_x_value_mechanical = np.loadtxt(
        "grad_shape_func_x_value_mechanical.txt"
    )
    grad_shape_func_y_value_mechanical = np.loadtxt(
        "grad_shape_func_y_value_mechanical.txt"
    )
    grad_shape_func_x_times_det_J_time_weight_value_mechanical = np.loadtxt(
        "grad_shape_func_x_times_det_J_time_weight_value_mechanical.txt.txt"
    )
    grad_shape_func_y_times_det_J_time_weight_value_mechanical = np.loadtxt(
        "grad_shape_func_y_times_det_J_time_weight_value_mechanical.txt.txt"
    )
    grad_shape_func_z_value_mechanical = np.loadtxt(
        "grad_shape_func_z_value_mechanical.txt"
    )
    grad_shape_func_z_times_det_J_time_weight_value_mechanical = np.loadtxt(
        "grad_shape_func_z_times_det_J_time_weight_value_mechanical.txt.txt"
    )

    # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
    shape_func_electrolyte = csc_matrix(
        (
            np.array(shape_func_value_electrolyte),
            (
                np.array(phi_nonzero_index_row_electrolyte),
                np.array(phi_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
    )
    shape_func_times_det_J_time_weight_electrolyte = csc_matrix(
        (
            np.array(shape_func_times_det_J_time_weight_value_electrolyte),
            (
                np.array(phi_nonzero_index_row_electrolyte),
                np.array(phi_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_x_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_x_value_electrolyte),
            (
                np.array(phi_nonzero_index_row_electrolyte),
                np.array(phi_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_y_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_y_value_electrolyte),
            (
                np.array(phi_nonzero_index_row_electrolyte),
                np.array(phi_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_x_times_det_J_time_weight_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_x_times_det_J_time_weight_value_electrolyte),
            (
                np.array(phi_nonzero_index_row_electrolyte),
                np.array(phi_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_y_times_det_J_time_weight_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_y_times_det_J_time_weight_value_electrolyte),
            (
                np.array(phi_nonzero_index_row_electrolyte),
                np.array(phi_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
    )

    shape_func_electrode = csc_matrix(
        (
            np.array(shape_func_value_electrode),
            (
                np.array(phi_nonzero_index_row_electrode),
                np.array(phi_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
    )
    shape_func_times_det_J_time_weight_electrode = csc_matrix(
        (
            np.array(shape_func_times_det_J_time_weight_value_electrode),
            (
                np.array(phi_nonzero_index_row_electrode),
                np.array(phi_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
    )
    grad_shape_func_x_electrode = csc_matrix(
        (
            np.array(grad_shape_func_x_value_electrode),
            (
                np.array(phi_nonzero_index_row_electrode),
                np.array(phi_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
    )
    grad_shape_func_y_electrode = csc_matrix(
        (
            np.array(grad_shape_func_y_value_electrode),
            (
                np.array(phi_nonzero_index_row_electrode),
                np.array(phi_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
    )
    grad_shape_func_x_times_det_J_time_weight_electrode = csc_matrix(
        (
            np.array(grad_shape_func_x_times_det_J_time_weight_value_electrode),
            (
                np.array(phi_nonzero_index_row_electrode),
                np.array(phi_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
    )
    grad_shape_func_y_times_det_J_time_weight_electrode = csc_matrix(
        (
            np.array(grad_shape_func_y_times_det_J_time_weight_value_electrode),
            (
                np.array(phi_nonzero_index_row_electrode),
                np.array(phi_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
    )

    shape_func_pore = csc_matrix(
        (
            np.array(shape_func_value_pore),
            (
                np.array(phi_nonzero_index_row_pore),
                np.array(phi_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
    )
    shape_func_times_det_J_time_weight_pore = csc_matrix(
        (
            np.array(shape_func_times_det_J_time_weight_value_pore),
            (
                np.array(phi_nonzero_index_row_pore),
                np.array(phi_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
    )
    grad_shape_func_x_pore = csc_matrix(
        (
            np.array(grad_shape_func_x_value_pore),
            (
                np.array(phi_nonzero_index_row_pore),
                np.array(phi_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
    )
    grad_shape_func_y_pore = csc_matrix(
        (
            np.array(grad_shape_func_y_value_pore),
            (
                np.array(phi_nonzero_index_row_pore),
                np.array(phi_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
    )
    grad_shape_func_x_times_det_J_time_weight_pore = csc_matrix(
        (
            np.array(grad_shape_func_x_times_det_J_time_weight_value_pore),
            (
                np.array(phi_nonzero_index_row_pore),
                np.array(phi_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
    )
    grad_shape_func_y_times_det_J_time_weight_pore = csc_matrix(
        (
            np.array(grad_shape_func_y_times_det_J_time_weight_value_pore),
            (
                np.array(phi_nonzero_index_row_pore),
                np.array(phi_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
    )

    shape_func_mechanical = csc_matrix(
        (
            np.array(shape_func_value_mechanical),
            (
                np.array(phi_nonzero_index_row_mechanical),
                np.array(phi_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
    )
    shape_func_times_det_J_time_weight_mechanical = csc_matrix(
        (
            np.array(shape_func_times_det_J_time_weight_value_mechanical),
            (
                np.array(phi_nonzero_index_row_mechanical),
                np.array(phi_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
    )
    grad_shape_func_x_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_x_value_mechanical),
            (
                np.array(phi_nonzero_index_row_mechanical),
                np.array(phi_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
    )
    grad_shape_func_y_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_y_value_mechanical),
            (
                np.array(phi_nonzero_index_row_mechanical),
                np.array(phi_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
    )
    grad_shape_func_x_times_det_J_time_weight_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_x_times_det_J_time_weight_value_mechanical),
            (
                np.array(phi_nonzero_index_row_mechanical),
                np.array(phi_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
    )
    grad_shape_func_y_times_det_J_time_weight_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_y_times_det_J_time_weight_value_mechanical),
            (
                np.array(phi_nonzero_index_row_mechanical),
                np.array(phi_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
    )

    if dimention == 3:
        grad_shape_func_z_electrolyte = csc_matrix(
            (
                np.array(grad_shape_func_z_value_electrolyte),
                (
                    np.array(phi_nonzero_index_row_electrolyte),
                    np.array(phi_nonzero_index_column_electrolyte),
                ),
            ),
            shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
        )
        grad_shape_func_z_times_det_J_time_weight_electrolyte = csc_matrix(
            (
                np.array(grad_shape_func_z_times_det_J_time_weight_value_electrolyte),
                (
                    np.array(phi_nonzero_index_row_electrolyte),
                    np.array(phi_nonzero_index_column_electrolyte),
                ),
            ),
            shape=(num_gauss_points_in_domain_electrolyte, num_nodes_electrolyte),
        )

        grad_shape_func_z_electrode = csc_matrix(
            (
                np.array(grad_shape_func_z_value_electrode),
                (
                    np.array(phi_nonzero_index_row_electrode),
                    np.array(phi_nonzero_index_column_electrode),
                ),
            ),
            shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
        )
        grad_shape_func_z_times_det_J_time_weight_electrode = csc_matrix(
            (
                np.array(grad_shape_func_z_times_det_J_time_weight_value_electrode),
                (
                    np.array(phi_nonzero_index_row_electrode),
                    np.array(phi_nonzero_index_column_electrode),
                ),
            ),
            shape=(num_gauss_points_in_domain_electrode, num_nodes_electrode),
        )

        grad_shape_func_z_pore = csc_matrix(
            (
                np.array(grad_shape_func_z_value_pore),
                (
                    np.array(phi_nonzero_index_row_pore),
                    np.array(phi_nonzero_index_column_pore),
                ),
            ),
            shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
        )
        grad_shape_func_z_times_det_J_time_weight_pore = csc_matrix(
            (
                np.array(grad_shape_func_z_times_det_J_time_weight_value_pore),
                (
                    np.array(phi_nonzero_index_row_pore),
                    np.array(phi_nonzero_index_column_pore),
                ),
            ),
            shape=(num_gauss_points_in_domain_pore, num_nodes_pore),
        )

        grad_shape_func_z_mechanical = csc_matrix(
            (
                np.array(grad_shape_func_z_value_mechanical),
                (
                    np.array(phi_nonzero_index_row_mechanical),
                    np.array(phi_nonzero_index_column_mechanical),
                ),
            ),
            shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
        )
        grad_shape_func_z_times_det_J_time_weight_mechanical = csc_matrix(
            (
                np.array(grad_shape_func_z_times_det_J_time_weight_value_mechanical),
                (
                    np.array(phi_nonzero_index_row_mechanical),
                    np.array(phi_nonzero_index_column_mechanical),
                ),
            ),
            shape=(num_gauss_points_in_domain_mechanical, num_nodes_mechanical),
        )

comp_shape_func_grad_shape_func_in_domain = time.time()

print(
    "time to compute the shape function and grad of shape function in domain = "
    + "%s seconds"
    % (comp_shape_func_grad_shape_func_in_domain - def_nodes_gauss_points_time)
)


#######################################################
# Compute shape function and its gradient on boundaries
########################################################

print("Compute shape function and its gradient on boundaries")

if studied_physics == "battery":
    M_b = np.array(
        [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary)],
        dtype=np.float64,
    )
    M_b_P_x = np.array(
        [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary)],
        dtype=np.float64,
    )
    M_b_P_y = np.array(
        [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary)],
        dtype=np.float64,
    )

    # save_point_D_coor_b, save_distance_function_b,save_distance_function_dx_b,save_distance_function_dy_b,
    (
        phi_b_nonzero_index_row,
        phi_b_nonzero_index_column,
        phi_b_nonzerovalue_data,
        phi_b_P_x_nonzerovalue_data,
        phi_b_P_y_nonzerovalue_data,
        phi_b_P_z_nonzerovalue_data,
        M_b,
        M_b_P_x,
        M_b_P_y,
        M_b_P_z,
    ) = compute_phi_M(
        x_G_b,
        Gauss_b_grain_id,
        x_nodes,
        nodes_grain_id,
        a,
        M_b,
        M_b_P_x,
        M_b_P_y,
        num_interface_segments,
        interface_nodes,
        BxByCxCy,
        IM_RKPM,
        single_grain,
    )

    num_non_zero_phi_a_b = np.shape(np.array(phi_b_nonzero_index_row))[0]

    (
        shape_func_b_value,
        shape_func_b_times_det_J_b_time_weight_value,
        grad_shape_func_b_x_value,
        grad_shape_func_b_y_value,
        grad_shape_func_b_z_value,
        grad_shape_func_b_x_times_det_J_b_time_weight_value,
        grad_shape_func_b_y_times_det_J_b_time_weight_value,
        grad_shape_func_b_z_times_det_J_b_time_weight_value,
    ) = shape_grad_shape_func(
        x_G_b,
        x_nodes,
        num_non_zero_phi_a_b,
        HT0,
        M_b,
        M_b_P_x,
        M_b_P_y,
        differential_method,
        HT1,
        HT2,
        phi_b_nonzerovalue_data,
        phi_b_P_x_nonzerovalue_data,
        phi_b_P_y_nonzerovalue_data,
        phi_b_nonzero_index_row,
        phi_b_nonzero_index_column,
        det_J_b_time_weight,
        IM_RKPM,
    )

    shape_func_b = csc_matrix(
        (
            np.array(shape_func_b_value),
            (np.array(phi_b_nonzero_index_row), np.array(phi_b_nonzero_index_column)),
        ),
        shape=(num_gauss_points_on_boundary, num_nodes),
    )
    shape_func_b_times_det_J_b_time_weight = csc_matrix(
        (
            np.array(shape_func_b_times_det_J_b_time_weight_value),
            (np.array(phi_b_nonzero_index_row), np.array(phi_b_nonzero_index_column)),
        ),
        shape=(num_gauss_points_on_boundary, num_nodes),
    )
    grad_shape_func_b_x = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value),
            (np.array(phi_b_nonzero_index_row), np.array(phi_b_nonzero_index_column)),
        ),
        shape=(num_gauss_points_on_boundary, num_nodes),
    )
    grad_shape_func_b_y = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value),
            (np.array(phi_b_nonzero_index_row), np.array(phi_b_nonzero_index_column)),
        ),
        shape=(num_gauss_points_on_boundary, num_nodes),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight = csc_matrix(
        (
            np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value),
            (np.array(phi_b_nonzero_index_row), np.array(phi_b_nonzero_index_column)),
        ),
        shape=(num_gauss_points_on_boundary, num_nodes),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight = csc_matrix(
        (
            np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value),
            (np.array(phi_b_nonzero_index_row), np.array(phi_b_nonzero_index_column)),
        ),
        shape=(num_gauss_points_on_boundary, num_nodes),
    )
else:
    if dimention == 2:
        M_b_electrolyte = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_x_electrolyte = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_y_electrolyte = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_electrolyte,
            phi_b_nonzero_index_column_electrolyte,
            phi_b_nonzerovalue_data_electrolyte,
            phi_b_P_x_nonzerovalue_data_electrolyte,
            phi_b_P_y_nonzerovalue_data_electrolyte,
            phi_b_P_z_nonzerovalue_data_electrolyte,
            M_b_electrolyte,
            M_b_P_x_electrolyte,
            M_b_P_y_electrolyte,
            M_b_P_z_electrolyte,
        ) = compute_phi_M(
            x_G_b_electrolyte,
            Gauss_b_grain_id_electrolyte,
            x_nodes_electrolyte,
            nodes_grain_id_electrolyte,
            a_electrolyte,
            M_b_electrolyte,
            M_b_P_x_electrolyte,
            M_b_P_y_electrolyte,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        M_b_mechanical = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_x_mechanical = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_y_mechanical = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )

        (
            phi_b_nonzero_index_row_mechanical,
            phi_b_nonzero_index_column_mechanical,
            phi_b_nonzerovalue_data_mechanical,
            phi_b_P_x_nonzerovalue_data_mechanical,
            phi_b_P_y_nonzerovalue_data_mechanical,
            phi_b_P_z_nonzerovalue_data_mechanical,
            M_b_mechanical,
            M_b_P_x_mechanical,
            M_b_P_y_mechanical,
            M_b_P_z_mechanical,
        ) = compute_phi_M(
            x_G_b_electrolyte,
            Gauss_b_grain_id_electrolyte,
            x_nodes_mechanical,
            nodes_grain_id_mechanical,
            a_mechanical,
            M_b_mechanical,
            M_b_P_x_mechanical,
            M_b_P_y_mechanical,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

    if dimention == 3:
        M_b_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_x_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_y_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_z_electrolyte = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_electrolyte, phi_b_nonzero_index_column_electrolyte, phi_b_nonzerovalue_data_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte,phi_b_P_z_nonzerovalue_data_electrolyte, M_b_electrolyte, M_b_P_x_electrolyte, M_b_P_y_electrolyte, M_b_P_z_electrolyte = compute_phi_M(x_G_b_electrolyte, Gauss_b_grain_id_electrolyte, x_nodes_electrolyte, nodes_grain_id_electrolyte, a_electrolyte, M_b_electrolyte, M_b_P_x_electrolyte, M_b_P_y_electrolyte, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrolyte)

        M_b_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_x_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_y_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        M_b_P_z_mechanical = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrolyte)],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_mechanical, phi_b_nonzero_index_column_mechanical, phi_b_nonzerovalue_data_mechanical, phi_b_P_x_nonzerovalue_data_mechanical, phi_b_P_y_nonzerovalue_data_mechanical,phi_b_P_z_nonzerovalue_data_mechanical, M_b_mechanical, M_b_P_x_mechanical, M_b_P_y_mechanical, M_b_P_z_mechanical = compute_phi_M(x_G_b_electrolyte, Gauss_b_grain_id_electrolyte, x_nodes_mechanical, nodes_grain_id_mechanical, a_mechanical, M_b_mechanical, M_b_P_x_mechanical, M_b_P_y_mechanical, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_mechanical)

    # np.savetxt('phi_b_nonzero_index_row_electrolyte.txt', phi_b_nonzero_index_row_electrolyte)
    # np.savetxt('phi_b_nonzero_index_column_electrolyte.txt', phi_b_nonzero_index_column_electrolyte)
    phi_b_nonzero_index_row_electrolyte = np.loadtxt(
        "phi_b_nonzero_index_row_electrolyte.txt"
    )
    phi_b_nonzero_index_column_electrolyte = np.loadtxt(
        "phi_b_nonzero_index_column_electrolyte.txt"
    )

    # np.savetxt('phi_b_nonzero_index_row_mechanical.txt', phi_b_nonzero_index_row_mechanical)
    # np.savetxt('phi_b_nonzero_index_column_mechanical.txt', phi_b_nonzero_index_column_mechanical)
    phi_b_nonzero_index_row_mechanical = np.loadtxt(
        "phi_b_nonzero_index_row_mechanical.txt"
    )
    phi_b_nonzero_index_column_mechanical = np.loadtxt(
        "phi_b_nonzero_index_column_mechanical.txt"
    )

    num_non_zero_phi_a_b_electrolyte = np.shape(
        np.array(phi_b_nonzero_index_row_electrolyte)
    )[0]
    num_non_zero_phi_a_b_mechanical = np.shape(
        np.array(phi_b_nonzero_index_row_mechanical)
    )[0]

    if dimention == 2:
        (
            shape_func_b_value_electrolyte,
            shape_func_b_times_det_J_b_time_weight_value_electrolyte,
            grad_shape_func_b_x_value_electrolyte,
            grad_shape_func_b_y_value_electrolyte,
            grad_shape_func_b_z_value_electrolyte,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte,
        ) = shape_grad_shape_func(
            x_G_b_electrolyte,
            x_nodes_electrolyte,
            num_non_zero_phi_a_b_electrolyte,
            HT0,
            M_b_electrolyte,
            M_b_P_x_electrolyte,
            M_b_P_y_electrolyte,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_electrolyte,
            phi_b_P_x_nonzerovalue_data_electrolyte,
            phi_b_P_y_nonzerovalue_data_electrolyte,
            phi_b_nonzero_index_row_electrolyte,
            phi_b_nonzero_index_column_electrolyte,
            det_J_b_time_weight_electrolyte,
            IM_RKPM,
        )
        (
            shape_func_b_value_mechanical,
            shape_func_b_times_det_J_b_time_weight_value_mechanical,
            grad_shape_func_b_x_value_mechanical,
            grad_shape_func_b_y_value_mechanical,
            grad_shape_func_b_z_value_mechanical,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical,
        ) = shape_grad_shape_func(
            x_G_b_electrolyte,
            x_nodes_mechanical,
            num_non_zero_phi_a_b_mechanical,
            HT0,
            M_b_mechanical,
            M_b_P_x_mechanical,
            M_b_P_y_mechanical,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_mechanical,
            phi_b_P_x_nonzerovalue_data_mechanical,
            phi_b_P_y_nonzerovalue_data_mechanical,
            phi_b_nonzero_index_row_mechanical,
            phi_b_nonzero_index_column_mechanical,
            det_J_b_time_weight_electrolyte,
            IM_RKPM,
        )

    if dimention == 3:
        print("yes")
        # shape_func_b_value_electrolyte, shape_func_b_times_det_J_b_time_weight_value_electrolyte, grad_shape_func_b_x_value_electrolyte, grad_shape_func_b_y_value_electrolyte,grad_shape_func_b_z_value_electrolyte, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte, grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte = shape_grad_shape_func(x_G_b_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_b_electrolyte,HT0, M_b_electrolyte, M_b_P_x_electrolyte, M_b_P_y_electrolyte, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte, phi_b_nonzero_index_row_electrolyte, phi_b_nonzero_index_column_electrolyte, det_J_b_time_weight_electrolyte, IM_RKPM, M_b_P_z_electrolyte, HT3, phi_b_P_z_nonzerovalue_data_electrolyte)
        # shape_func_b_value_mechanical, shape_func_b_times_det_J_b_time_weight_value_mechanical, grad_shape_func_b_x_value_mechanical, grad_shape_func_b_y_value_mechanical,grad_shape_func_b_z_value_mechanical, grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical, grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical, grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical = shape_grad_shape_func(x_G_b_electrolyte,x_nodes_mechanical, num_non_zero_phi_a_b_mechanical,HT0, M_b_mechanical, M_b_P_x_mechanical, M_b_P_y_mechanical, differential_method, HT1, HT2, phi_b_nonzerovalue_data_mechanical, phi_b_P_x_nonzerovalue_data_mechanical, phi_b_P_y_nonzerovalue_data_mechanical, phi_b_nonzero_index_row_mechanical, phi_b_nonzero_index_column_mechanical, det_J_b_time_weight_electrolyte, IM_RKPM, M_b_P_z_mechanical, HT3, phi_b_P_z_nonzerovalue_data_mechanical)

    # np.savetxt('shape_func_b_value_electrolyte.txt', np.asarray(shape_func_b_value_electrolyte))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_electrolyte.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_electrolyte))
    # np.savetxt('grad_shape_func_b_x_value_electrolyte.txt', np.asarray(grad_shape_func_b_x_value_electrolyte))
    # np.savetxt('grad_shape_func_b_y_value_electrolyte.txt', np.asarray(grad_shape_func_b_y_value_electrolyte))
    # np.savetxt('grad_shape_func_b_z_value_electrolyte.txt', np.asarray(grad_shape_func_b_z_value_electrolyte))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte))
    shape_func_b_value_electrolyte = np.loadtxt("shape_func_b_value_electrolyte.txt")
    shape_func_b_times_det_J_b_time_weight_value_electrolyte = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_electrolyte.txt"
    )
    grad_shape_func_b_x_value_electrolyte = np.loadtxt(
        "grad_shape_func_b_x_value_electrolyte.txt"
    )
    grad_shape_func_b_y_value_electrolyte = np.loadtxt(
        "grad_shape_func_b_y_value_electrolyte.txt"
    )
    grad_shape_func_b_z_value_electrolyte = np.loadtxt(
        "grad_shape_func_b_z_value_electrolyte.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte.txt"
    )

    # np.savetxt('shape_func_b_value_mechanical.txt', np.asarray(shape_func_b_value_mechanical))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_mechanical.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_mechanical))
    # np.savetxt('grad_shape_func_b_x_value_mechanical.txt', np.asarray(grad_shape_func_b_x_value_mechanical))
    # np.savetxt('grad_shape_func_b_y_value_mechanical.txt', np.asarray(grad_shape_func_b_y_value_mechanical))
    # np.savetxt('grad_shape_func_b_z_value_mechanical.txt', np.asarray(grad_shape_func_b_z_value_mechanical))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical))
    shape_func_b_value_mechanical = np.loadtxt("shape_func_b_value_electrolyte.txt")
    shape_func_b_times_det_J_b_time_weight_value_mechanical = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_mechanical.txt"
    )
    grad_shape_func_b_x_value_mechanical = np.loadtxt(
        "grad_shape_func_b_x_value_mechanical.txt"
    )
    grad_shape_func_b_y_value_mechanical = np.loadtxt(
        "grad_shape_func_b_y_value_mechanical.txt"
    )
    grad_shape_func_b_z_value_mechanical = np.loadtxt(
        "grad_shape_func_b_z_value_mechanical.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical.txt"
    )

    shape_func_b_electrolyte = csc_matrix(
        (
            np.array(shape_func_b_value_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
    )
    shape_func_b_times_det_J_b_time_weight_electrolyte = csc_matrix(
        (
            np.array(shape_func_b_times_det_J_b_time_weight_value_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_b_x_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_b_y_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
    )

    shape_func_b_mechanical = csc_matrix(
        (
            np.array(shape_func_b_value_mechanical),
            (
                np.array(phi_b_nonzero_index_row_mechanical),
                np.array(phi_b_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
    )
    shape_func_b_times_det_J_b_time_weight_mechanical = csc_matrix(
        (
            np.array(shape_func_b_times_det_J_b_time_weight_value_mechanical),
            (
                np.array(phi_b_nonzero_index_row_mechanical),
                np.array(phi_b_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
    )
    grad_shape_func_b_x_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_mechanical),
            (
                np.array(phi_b_nonzero_index_row_mechanical),
                np.array(phi_b_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
    )
    grad_shape_func_b_y_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_mechanical),
            (
                np.array(phi_b_nonzero_index_row_mechanical),
                np.array(phi_b_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_mechanical),
            (
                np.array(phi_b_nonzero_index_row_mechanical),
                np.array(phi_b_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_mechanical = csc_matrix(
        (
            np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_mechanical),
            (
                np.array(phi_b_nonzero_index_row_mechanical),
                np.array(phi_b_nonzero_index_column_mechanical),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
    )

    if dimention == 3:
        grad_shape_func_b_z_electrolyte = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_electrolyte),
                (
                    np.array(phi_b_nonzero_index_row_electrolyte),
                    np.array(phi_b_nonzero_index_column_electrolyte),
                ),
            ),
            shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte = csc_matrix(
            (
                np.array(
                    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte
                ),
                (
                    np.array(phi_b_nonzero_index_row_electrolyte),
                    np.array(phi_b_nonzero_index_column_electrolyte),
                ),
            ),
            shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_electrolyte),
        )

        grad_shape_func_b_z_mechanical = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_mechanical),
                (
                    np.array(phi_b_nonzero_index_row_mechanical),
                    np.array(phi_b_nonzero_index_column_mechanical),
                ),
            ),
            shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_mechanical = csc_matrix(
            (
                np.array(
                    grad_shape_func_b_z_times_det_J_b_time_weight_value_mechanical
                ),
                (
                    np.array(phi_b_nonzero_index_row_mechanical),
                    np.array(phi_b_nonzero_index_column_mechanical),
                ),
            ),
            shape=(num_gauss_points_on_boundary_electrolyte, num_nodes_mechanical),
        )

    if dimention == 2:
        M_b_electrode = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        M_b_P_x_electrode = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        M_b_P_y_electrode = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_electrode,
            phi_b_nonzero_index_column_electrode,
            phi_b_nonzerovalue_data_electrode,
            phi_b_P_x_nonzerovalue_data_electrode,
            phi_b_P_y_nonzerovalue_data_electrode,
            phi_b_P_z_nonzerovalue_data_electrode,
            M_b_electrode,
            M_b_P_x_electrode,
            M_b_P_y_electrode,
            M_b_P_z_electrode,
        ) = compute_phi_M(
            x_G_b_electrode,
            Gauss_b_grain_id_electrode,
            x_nodes_electrode,
            nodes_grain_id_electrode,
            a_electrode,
            M_b_electrode,
            M_b_P_x_electrode,
            M_b_P_y_electrode,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_b_pore = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        M_b_P_x_pore = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        M_b_P_y_pore = np.array(
            [np.zeros((3, 3)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_pore,
            phi_b_nonzero_index_column_pore,
            phi_b_nonzerovalue_data_pore,
            phi_b_P_x_nonzerovalue_data_pore,
            phi_b_P_y_nonzerovalue_data_pore,
            phi_b_P_z_nonzerovalue_data_pore,
            M_b_pore,
            M_b_P_x_pore,
            M_b_P_y_pore,
            M_b_P_z_pore,
        ) = compute_phi_M(
            x_G_b_pore,
            Gauss_b_grain_id_pore,
            x_nodes_pore,
            nodes_grain_id_pore,
            a_pore,
            M_b_pore,
            M_b_P_x_pore,
            M_b_P_y_pore,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_b_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_electrolyte_electrode_electrolyte,
            phi_b_nonzero_index_column_electrolyte_electrode_electrolyte,
            phi_b_nonzerovalue_data_electrolyte_electrode_electrolyte,
            phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrolyte,
            phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrolyte,
            phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrolyte,
            M_b_electrolyte_electrode_electrolyte,
            M_b_P_x_electrolyte_electrode_electrolyte,
            M_b_P_y_electrolyte_electrode_electrolyte,
            M_b_P_z_electrolyte_electrode_electrolyte,
        ) = compute_phi_M(
            x_G_b_interface_electrode_electrolyte,
            Gauss_b_grain_id_electrolyte_electrode_interace,
            x_nodes_electrolyte,
            nodes_grain_id_electrolyte,
            a_electrolyte,
            M_b_electrolyte_electrode_electrolyte,
            M_b_P_x_electrolyte_electrode_electrolyte,
            M_b_P_y_electrolyte_electrode_electrolyte,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_b_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_electrolyte_electrode_electrode,
            phi_b_nonzero_index_column_electrolyte_electrode_electrode,
            phi_b_nonzerovalue_data_electrolyte_electrode_electrode,
            phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrode,
            phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrode,
            phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrode,
            M_b_electrolyte_electrode_electrode,
            M_b_P_x_electrolyte_electrode_electrode,
            M_b_P_y_electrolyte_electrode_electrode,
            M_b_P_z_electrolyte_electrode_electrode,
        ) = compute_phi_M(
            x_G_b_interface_electrode_electrolyte,
            Gauss_b_grain_id_electrolyte_electrode_interace,
            x_nodes_electrode,
            nodes_grain_id_electrode,
            a_electrode,
            M_b_electrolyte_electrode_electrode,
            M_b_P_x_electrolyte_electrode_electrode,
            M_b_P_y_electrolyte_electrode_electrode,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_b_electrode_pore_electrode = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrode_pore_electrode = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrode_pore_electrode = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_electrode_pore_electrode,
            phi_b_nonzero_index_column_electrode_pore_electrode,
            phi_b_nonzerovalue_data_electrode_pore_electrode,
            phi_b_P_x_nonzerovalue_data_electrode_pore_electrode,
            phi_b_P_y_nonzerovalue_data_electrode_pore_electrode,
            phi_b_P_z_nonzerovalue_data_electrode_pore_electrode,
            M_b_electrode_pore_electrode,
            M_b_P_x_electrode_pore_electrode,
            M_b_P_y_electrode_pore_electrode,
            M_b_P_z_electrode_pore_electrode,
        ) = compute_phi_M(
            x_G_b_interface_electrode_pore,
            Gauss_b_grain_id_electrode_pore_interace,
            x_nodes_electrode,
            nodes_grain_id_electrode,
            a_electrode,
            M_b_electrode_pore_electrode,
            M_b_P_x_electrode_pore_electrode,
            M_b_P_y_electrode_pore_electrode,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_b_electrode_pore_pore = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrode_pore_pore = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrode_pore_pore = np.array(
            [
                np.zeros((3, 3))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        (
            phi_b_nonzero_index_row_electrode_pore_pore,
            phi_b_nonzero_index_column_electrode_pore_pore,
            phi_b_nonzerovalue_data_electrode_pore_pore,
            phi_b_P_x_nonzerovalue_data_electrode_pore_pore,
            phi_b_P_y_nonzerovalue_data_electrode_pore_pore,
            phi_b_P_z_nonzerovalue_data_electrode_pore_pore,
            M_b_electrode_pore_pore,
            M_b_P_x_electrode_pore_pore,
            M_b_P_y_electrode_pore_pore,
            M_b_P_z_electrode_pore_pore,
        ) = compute_phi_M(
            x_G_b_interface_electrode_pore,
            Gauss_b_grain_id_electrode_pore_interace,
            x_nodes_pore,
            nodes_grain_id_pore,
            a_pore,
            M_b_electrode_pore_pore,
            M_b_P_x_electrode_pore_pore,
            M_b_P_y_electrode_pore_pore,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        if delta_point_source == "False":
            print(np.shape(x_G_b_distributed_point_source_line)[0])
            # exit()
            M_b_distributed_point_source_line = np.array(
                [
                    np.zeros((3, 3))
                    for _ in range(np.shape(x_G_b_distributed_point_source_line)[0])
                ],
                dtype=np.float64,
            )
            M_b_P_x_distributed_point_source_line = np.array(
                [
                    np.zeros((3, 3))
                    for _ in range(np.shape(x_G_b_distributed_point_source_line)[0])
                ],
                dtype=np.float64,
            )
            M_b_P_y_distributed_point_source_line = np.array(
                [
                    np.zeros((3, 3))
                    for _ in range(np.shape(x_G_b_distributed_point_source_line)[0])
                ],
                dtype=np.float64,
            )
            Gauss_b_grain_id_distributed_point_source_line = 1 * np.ones(
                np.shape(x_G_b_distributed_point_source_line)[0]
            )
            (
                phi_b_nonzero_index_row_distributed_point_source_line,
                phi_b_nonzero_index_column_distributed_point_source_line,
                phi_b_nonzerovalue_data_distributed_point_source_line,
                phi_b_P_x_nonzerovalue_data_distributed_point_source_line,
                phi_b_P_y_nonzerovalue_data_distributed_point_source_line,
                phi_b_P_z_nonzerovalue_data_distributed_point_source_line,
                M_b_distributed_point_source_line,
                M_b_P_x_distributed_point_source_line,
                M_b_P_y_distributed_point_source_line,
                M_b_P_z_distributed_point_source_line,
            ) = compute_phi_M(
                np.array(x_G_b_distributed_point_source_line),
                Gauss_b_grain_id_distributed_point_source_line,
                x_nodes_electrolyte,
                nodes_grain_id_electrolyte,
                a_electrolyte,
                M_b_distributed_point_source_line,
                M_b_P_x_distributed_point_source_line,
                M_b_P_y_distributed_point_source_line,
                num_interface_segments,
                interface_nodes,
                BxByCxCy,
                IM_RKPM,
                single_grain,
            )

            np.savetxt(
                "phi_b_nonzero_index_row_distributed_point_source_line.txt",
                phi_b_nonzero_index_row_distributed_point_source_line,
            )
            np.savetxt(
                "phi_b_nonzero_index_column_distributed_point_source_line.txt",
                phi_b_nonzero_index_column_distributed_point_source_line,
            )
            phi_b_nonzero_index_row_distributed_point_source_line = np.loadtxt(
                "phi_b_nonzero_index_row_distributed_point_source_line.txt"
            )
            phi_b_nonzero_index_column_distributed_point_source_line = np.loadtxt(
                "phi_b_nonzero_index_column_distributed_point_source_line.txt"
            )
            num_non_zero_phi_a_b_distributed_point_source_line = np.shape(
                np.array(phi_b_nonzero_index_row_distributed_point_source_line)
            )[0]

    if dimention == 3:
        M_b_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        M_b_P_x_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        M_b_P_y_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        M_b_P_z_electrode = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_electrode)],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_electrode, phi_b_nonzero_index_column_electrode, phi_b_nonzerovalue_data_electrode, phi_b_P_x_nonzerovalue_data_electrode, phi_b_P_y_nonzerovalue_data_electrode,phi_b_P_z_nonzerovalue_data_electrode, M_b_electrode, M_b_P_x_electrode, M_b_P_y_electrode, M_b_P_z_electrode = compute_phi_M(x_G_b_electrode, Gauss_b_grain_id_electrode, x_nodes_electrode, nodes_grain_id_electrode, a_electrode, M_b_electrode, M_b_P_x_electrode, M_b_P_y_electrode, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrode)

        M_b_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        M_b_P_x_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        M_b_P_y_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        M_b_P_z_pore = np.array(
            [np.zeros((4, 4)) for _ in range(num_gauss_points_on_boundary_pore)],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_pore, phi_b_nonzero_index_column_pore, phi_b_nonzerovalue_data_pore, phi_b_P_x_nonzerovalue_data_pore, phi_b_P_y_nonzerovalue_data_pore,phi_b_P_z_nonzerovalue_data_pore, M_b_pore, M_b_P_x_pore, M_b_P_y_pore, M_b_P_z_pore = compute_phi_M(x_G_b_pore, Gauss_b_grain_id_pore, x_nodes_pore, nodes_grain_id_pore, a_pore, M_b_pore, M_b_P_x_pore, M_b_P_y_pore, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_pore)

        M_b_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_z_electrolyte_electrode_electrolyte = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_electrolyte_electrode_electrolyte, phi_b_nonzero_index_column_electrolyte_electrode_electrolyte, phi_b_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrolyte,phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrolyte, M_b_electrolyte_electrode_electrolyte, M_b_P_x_electrolyte_electrode_electrolyte, M_b_P_y_electrolyte_electrode_electrolyte, M_b_P_z_electrolyte_electrode_electrolyte = compute_phi_M(x_G_b_interface_electrode_electrolyte, Gauss_b_grain_id_electrolyte_electrode_interace, x_nodes_electrolyte, nodes_grain_id_electrolyte, a_electrolyte, M_b_electrolyte_electrode_electrolyte, M_b_P_x_electrolyte_electrode_electrolyte, M_b_P_y_electrolyte_electrode_electrolyte, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrolyte_electrode_electrolyte)

        M_b_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_z_electrolyte_electrode_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrolyte_electrode_interface)
            ],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_electrolyte_electrode_electrode, phi_b_nonzero_index_column_electrolyte_electrode_electrode, phi_b_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrode,phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrode, M_b_electrolyte_electrode_electrode, M_b_P_x_electrolyte_electrode_electrode, M_b_P_y_electrolyte_electrode_electrode, M_b_P_z_electrolyte_electrode_electrode = compute_phi_M(x_G_b_interface_electrode_electrolyte, Gauss_b_grain_id_electrolyte_electrode_interace, x_nodes_electrode, nodes_grain_id_electrode, a_electrode, M_b_electrolyte_electrode_electrode, M_b_P_x_electrolyte_electrode_electrode, M_b_P_y_electrolyte_electrode_electrode, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrolyte_electrode_electrode)

        M_b_electrode_pore_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrode_pore_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrode_pore_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_z_electrode_pore_electrode = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_electrode_pore_electrode, phi_b_nonzero_index_column_electrode_pore_electrode, phi_b_nonzerovalue_data_electrode_pore_electrode, phi_b_P_x_nonzerovalue_data_electrode_pore_electrode, phi_b_P_y_nonzerovalue_data_electrode_pore_electrode,phi_b_P_z_nonzerovalue_data_electrode_pore_electrode, M_b_electrode_pore_electrode, M_b_P_x_electrode_pore_electrode, M_b_P_y_electrode_pore_electrode, M_b_P_z_electrode_pore_electrode = compute_phi_M(x_G_b_interface_electrode_pore, Gauss_b_grain_id_electrode_pore_interace, x_nodes_electrode, nodes_grain_id_electrode, a_electrode, M_b_electrode_pore_electrode, M_b_P_x_electrode_pore_electrode, M_b_P_y_electrode_pore_electrode, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrode_pore_electrode)

        M_b_electrode_pore_pore = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_x_electrode_pore_pore = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_y_electrode_pore_pore = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        M_b_P_z_electrode_pore_pore = np.array(
            [
                np.zeros((4, 4))
                for _ in range(num_gauss_points_on_electrode_pore_interface)
            ],
            dtype=np.float64,
        )
        # phi_b_nonzero_index_row_electrode_pore_pore, phi_b_nonzero_index_column_electrode_pore_pore, phi_b_nonzerovalue_data_electrode_pore_pore, phi_b_P_x_nonzerovalue_data_electrode_pore_pore, phi_b_P_y_nonzerovalue_data_electrode_pore_pore,phi_b_P_z_nonzerovalue_data_electrode_pore_pore, M_b_electrode_pore_pore, M_b_P_x_electrode_pore_pore, M_b_P_y_electrode_pore_pore, M_b_P_z_electrode_pore_pore = compute_phi_M(x_G_b_interface_electrode_pore, Gauss_b_grain_id_electrode_pore_interace, x_nodes_pore, nodes_grain_id_pore, a_pore, M_b_electrode_pore_pore, M_b_P_x_electrode_pore_pore, M_b_P_y_electrode_pore_pore, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_electrode_pore_pore)

        if delta_point_source == "False":
            print(np.shape(x_G_b_distributed_point_source_surface)[0])
            M_b_distributed_point_source_surface = np.array(
                [
                    np.zeros((4, 4))
                    for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])
                ],
                dtype=np.float64,
            )
            M_b_P_x_distributed_point_source_surface = np.array(
                [
                    np.zeros((4, 4))
                    for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])
                ],
                dtype=np.float64,
            )
            M_b_P_y_distributed_point_source_surface = np.array(
                [
                    np.zeros((4, 4))
                    for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])
                ],
                dtype=np.float64,
            )
            M_b_P_z_distributed_point_source_surface = np.array(
                [
                    np.zeros((4, 4))
                    for _ in range(np.shape(x_G_b_distributed_point_source_surface)[0])
                ],
                dtype=np.float64,
            )
            Gauss_b_grain_id_distributed_point_source_surface = 1 * np.ones(
                np.shape(x_G_b_distributed_point_source_surface)[0]
            )
            # phi_b_nonzero_index_row_distributed_point_source_surface, phi_b_nonzero_index_column_distributed_point_source_surface, phi_b_nonzerovalue_data_distributed_point_source_surface, phi_b_P_x_nonzerovalue_data_distributed_point_source_surface, phi_b_P_y_nonzerovalue_data_distributed_point_source_surface,phi_b_P_z_nonzerovalue_data_distributed_point_source_surface, M_b_distributed_point_source_surface, M_b_P_x_distributed_point_source_surface, M_b_P_y_distributed_point_source_surface, M_b_P_z_distributed_point_source_surface = compute_phi_M(np.array(x_G_b_distributed_point_source_surface), Gauss_b_grain_id_distributed_point_source_surface, x_nodes_electrolyte, nodes_grain_id_electrolyte, a_electrolyte, M_b_distributed_point_source_surface, M_b_P_x_distributed_point_source_surface, M_b_P_y_distributed_point_source_surface, num_interface_segments, interface_nodes, BxByCxCy,IM_RKPM, single_grain, M_b_P_z_distributed_point_source_surface)

            # np.savetxt('phi_b_nonzero_index_row_distributed_point_source_surface.txt', phi_b_nonzero_index_row_distributed_point_source_surface)
            # np.savetxt('phi_b_nonzero_index_column_distributed_point_source_surface.txt', phi_b_nonzero_index_column_distributed_point_source_surface)
            phi_b_nonzero_index_row_distributed_point_source_surface = np.loadtxt(
                "phi_b_nonzero_index_row_distributed_point_source_surface.txt"
            )
            phi_b_nonzero_index_column_distributed_point_source_surface = np.loadtxt(
                "phi_b_nonzero_index_column_distributed_point_source_surface.txt"
            )
            num_non_zero_phi_a_b_distributed_point_source_surface = np.shape(
                np.array(phi_b_nonzero_index_row_distributed_point_source_surface)
            )[0]

    # np.savetxt('phi_b_nonzero_index_row_electrode.txt', phi_b_nonzero_index_row_electrode)
    # np.savetxt('phi_b_nonzero_index_column_electrode.txt', phi_b_nonzero_index_column_electrode)
    phi_b_nonzero_index_row_electrode = np.loadtxt(
        "phi_b_nonzero_index_row_electrode.txt"
    )
    phi_b_nonzero_index_column_electrode = np.loadtxt(
        "phi_b_nonzero_index_column_electrode.txt"
    )

    # np.savetxt('phi_b_nonzero_index_row_pore.txt', phi_b_nonzero_index_row_pore)
    # np.savetxt('phi_b_nonzero_index_column_pore.txt', phi_b_nonzero_index_column_pore)
    phi_b_nonzero_index_row_pore = np.loadtxt("phi_b_nonzero_index_row_pore.txt")
    phi_b_nonzero_index_column_pore = np.loadtxt("phi_b_nonzero_index_column_pore.txt")

    # np.savetxt('phi_b_nonzero_index_row_electrolyte_electrode_electrolyte.txt', phi_b_nonzero_index_row_electrolyte_electrode_electrolyte)
    # np.savetxt('phi_b_nonzero_index_column_electrolyte_electrode_electrolyte.txt', phi_b_nonzero_index_column_electrolyte_electrode_electrolyte)
    phi_b_nonzero_index_row_electrolyte_electrode_electrolyte = np.loadtxt(
        "phi_b_nonzero_index_row_electrolyte_electrode_electrolyte.txt"
    )
    phi_b_nonzero_index_column_electrolyte_electrode_electrolyte = np.loadtxt(
        "phi_b_nonzero_index_column_electrolyte_electrode_electrolyte.txt"
    )

    # np.savetxt('phi_b_nonzero_index_row_electrolyte_electrode_electrode.txt', phi_b_nonzero_index_row_electrolyte_electrode_electrode)
    # np.savetxt('phi_b_nonzero_index_column_electrolyte_electrode_electrode.txt', phi_b_nonzero_index_column_electrolyte_electrode_electrode)
    phi_b_nonzero_index_row_electrolyte_electrode_electrode = np.loadtxt(
        "phi_b_nonzero_index_row_electrolyte_electrode_electrode.txt"
    )
    phi_b_nonzero_index_column_electrolyte_electrode_electrode = np.loadtxt(
        "phi_b_nonzero_index_column_electrolyte_electrode_electrode.txt"
    )

    # np.savetxt('phi_b_nonzero_index_row_electrode_pore_electrode.txt', phi_b_nonzero_index_row_electrode_pore_electrode)
    # np.savetxt('phi_b_nonzero_index_column_electrode_pore_electrode.txt', phi_b_nonzero_index_column_electrode_pore_electrode)
    phi_b_nonzero_index_row_electrode_pore_electrode = np.loadtxt(
        "phi_b_nonzero_index_row_electrode_pore_electrode.txt"
    )
    phi_b_nonzero_index_column_electrode_pore_electrode = np.loadtxt(
        "phi_b_nonzero_index_column_electrode_pore_electrode.txt"
    )

    # np.savetxt('phi_b_nonzero_index_row_electrode_pore_pore.txt', phi_b_nonzero_index_row_electrode_pore_pore)
    # np.savetxt('phi_b_nonzero_index_column_electrode_pore_pore.txt', phi_b_nonzero_index_column_electrode_pore_pore)
    phi_b_nonzero_index_row_electrode_pore_pore = np.loadtxt(
        "phi_b_nonzero_index_row_electrode_pore_pore.txt"
    )
    phi_b_nonzero_index_column_electrode_pore_pore = np.loadtxt(
        "phi_b_nonzero_index_column_electrode_pore_pore.txt"
    )

    num_non_zero_phi_a_b_electrode = np.shape(
        np.array(phi_b_nonzero_index_row_electrode)
    )[0]
    num_non_zero_phi_a_b_pore = np.shape(np.array(phi_b_nonzero_index_row_pore))[0]
    num_non_zero_phi_a_b_electrolyte_electrode_electrolyte = np.shape(
        np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte)
    )[0]
    num_non_zero_phi_a_b_electrolyte_electrode_electrode = np.shape(
        np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode)
    )[0]
    num_non_zero_phi_a_b_electrode_pore_electrode = np.shape(
        np.array(phi_b_nonzero_index_row_electrode_pore_electrode)
    )[0]
    num_non_zero_phi_a_b_electrode_pore_pore = np.shape(
        np.array(phi_b_nonzero_index_row_electrode_pore_pore)
    )[0]

    if dimention == 2:
        (
            shape_func_b_value_electrode,
            shape_func_b_times_det_J_b_time_weight_value_electrode,
            grad_shape_func_b_x_value_electrode,
            grad_shape_func_b_y_value_electrode,
            grad_shape_func_b_z_value_electrode,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode,
        ) = shape_grad_shape_func(
            x_G_b_electrode,
            x_nodes_electrode,
            num_non_zero_phi_a_b_electrode,
            HT0,
            M_b_electrode,
            M_b_P_x_electrode,
            M_b_P_y_electrode,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_electrode,
            phi_b_P_x_nonzerovalue_data_electrode,
            phi_b_P_y_nonzerovalue_data_electrode,
            phi_b_nonzero_index_row_electrode,
            phi_b_nonzero_index_column_electrode,
            det_J_b_time_weight_electrode,
            IM_RKPM,
        )

        (
            shape_func_b_value_pore,
            shape_func_b_times_det_J_b_time_weight_value_pore,
            grad_shape_func_b_x_value_pore,
            grad_shape_func_b_y_value_pore,
            grad_shape_func_b_z_value_pore,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_pore,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_pore,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_pore,
        ) = shape_grad_shape_func(
            x_G_b_pore,
            x_nodes_pore,
            num_non_zero_phi_a_b_pore,
            HT0,
            M_b_pore,
            M_b_P_x_pore,
            M_b_P_y_pore,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_pore,
            phi_b_P_x_nonzerovalue_data_pore,
            phi_b_P_y_nonzerovalue_data_pore,
            phi_b_nonzero_index_row_pore,
            phi_b_nonzero_index_column_pore,
            det_J_b_time_weight_pore,
            IM_RKPM,
        )

        (
            shape_func_b_value_electrolyte_electrode_electrolyte,
            shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte,
            grad_shape_func_b_x_value_electrolyte_electrode_electrolyte,
            grad_shape_func_b_y_value_electrolyte_electrode_electrolyte,
            grad_shape_func_b_z_value_electrolyte_electrode_electrolyte,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte,
        ) = shape_grad_shape_func(
            x_G_b_interface_electrode_electrolyte,
            x_nodes_electrolyte,
            num_non_zero_phi_a_b_electrolyte_electrode_electrolyte,
            HT0,
            M_b_electrolyte_electrode_electrolyte,
            M_b_P_x_electrolyte_electrode_electrolyte,
            M_b_P_y_electrolyte_electrode_electrolyte,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_electrolyte_electrode_electrolyte,
            phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrolyte,
            phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrolyte,
            phi_b_nonzero_index_row_electrolyte_electrode_electrolyte,
            phi_b_nonzero_index_column_electrolyte_electrode_electrolyte,
            det_J_b_time_weight_interface_electrode_electrolyte,
            IM_RKPM,
        )

        (
            shape_func_b_value_electrolyte_electrode_electrode,
            shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode,
            grad_shape_func_b_x_value_electrolyte_electrode_electrode,
            grad_shape_func_b_y_value_electrolyte_electrode_electrode,
            grad_shape_func_b_z_value_electrolyte_electrode_electrode,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode,
        ) = shape_grad_shape_func(
            x_G_b_interface_electrode_electrolyte,
            x_nodes_electrode,
            num_non_zero_phi_a_b_electrolyte_electrode_electrode,
            HT0,
            M_b_electrolyte_electrode_electrode,
            M_b_P_x_electrolyte_electrode_electrode,
            M_b_P_y_electrolyte_electrode_electrode,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_electrolyte_electrode_electrode,
            phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrode,
            phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrode,
            phi_b_nonzero_index_row_electrolyte_electrode_electrode,
            phi_b_nonzero_index_column_electrolyte_electrode_electrode,
            det_J_b_time_weight_interface_electrode_electrolyte,
            IM_RKPM,
        )

        (
            shape_func_b_value_electrode_pore_electrode,
            shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode,
            grad_shape_func_b_x_value_electrode_pore_electrode,
            grad_shape_func_b_y_value_electrode_pore_electrode,
            grad_shape_func_b_z_value_electrode_pore_electrode,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode,
        ) = shape_grad_shape_func(
            x_G_b_interface_electrode_pore,
            x_nodes_electrode,
            num_non_zero_phi_a_b_electrode_pore_electrode,
            HT0,
            M_b_electrode_pore_electrode,
            M_b_P_x_electrode_pore_electrode,
            M_b_P_y_electrode_pore_electrode,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_electrode_pore_electrode,
            phi_b_P_x_nonzerovalue_data_electrode_pore_electrode,
            phi_b_P_y_nonzerovalue_data_electrode_pore_electrode,
            phi_b_nonzero_index_row_electrode_pore_electrode,
            phi_b_nonzero_index_column_electrode_pore_electrode,
            det_J_b_time_weight_interface_electrode_pore,
            IM_RKPM,
        )

        (
            shape_func_b_value_electrode_pore_pore,
            shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore,
            grad_shape_func_b_x_value_electrode_pore_pore,
            grad_shape_func_b_y_value_electrode_pore_pore,
            grad_shape_func_b_z_value_electrode_pore_pore,
            grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore,
            grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore,
            grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore,
        ) = shape_grad_shape_func(
            x_G_b_interface_electrode_pore,
            x_nodes_pore,
            num_non_zero_phi_a_b_electrode_pore_pore,
            HT0,
            M_b_electrode_pore_pore,
            M_b_P_x_electrode_pore_pore,
            M_b_P_y_electrode_pore_pore,
            differential_method,
            HT1,
            HT2,
            phi_b_nonzerovalue_data_electrode_pore_pore,
            phi_b_P_x_nonzerovalue_data_electrode_pore_pore,
            phi_b_P_y_nonzerovalue_data_electrode_pore_pore,
            phi_b_nonzero_index_row_electrode_pore_pore,
            phi_b_nonzero_index_column_electrode_pore_pore,
            det_J_b_time_weight_interface_electrode_pore,
            IM_RKPM,
        )

        if delta_point_source == "False":
            num_gauss_points_distributed_point_source_line = np.shape(
                x_G_b_distributed_point_source_line
            )[0]
            (
                shape_func_b_value_distributed_point_source_line,
                shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_line,
                grad_shape_func_b_x_value_distributed_point_source_line,
                grad_shape_func_b_y_value_distributed_point_source_line,
                grad_shape_func_b_z_value_distributed_point_source_line,
                grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_line,
                grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_line,
                grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_line,
            ) = shape_grad_shape_func(
                x_G_b_distributed_point_source_line,
                x_nodes_electrolyte,
                num_non_zero_phi_a_b_distributed_point_source_line,
                HT0,
                M_b_distributed_point_source_line,
                M_b_P_x_distributed_point_source_line,
                M_b_P_y_distributed_point_source_line,
                differential_method,
                HT1,
                HT2,
                phi_b_nonzerovalue_data_distributed_point_source_line,
                phi_b_P_x_nonzerovalue_data_distributed_point_source_line,
                phi_b_P_y_nonzerovalue_data_distributed_point_source_line,
                phi_b_nonzero_index_row_distributed_point_source_line,
                phi_b_nonzero_index_column_distributed_point_source_line,
                det_J_b_time_weight_distributed_point_source_line,
                IM_RKPM,
            )
            shape_func_b_distributed_point_source_line = csc_matrix(
                (
                    np.array(shape_func_b_value_distributed_point_source_line),
                    (
                        np.array(phi_b_nonzero_index_row_distributed_point_source_line),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_line
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_line,
                    num_nodes_electrolyte,
                ),
            )
            shape_func_b_times_det_J_b_time_weight_distributed_point_source_line = csc_matrix(
                (
                    np.array(
                        shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_line
                    ),
                    (
                        np.array(phi_b_nonzero_index_row_distributed_point_source_line),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_line
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_line,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_x_distributed_point_source_line = csc_matrix(
                (
                    np.array(grad_shape_func_b_x_value_distributed_point_source_line),
                    (
                        np.array(phi_b_nonzero_index_row_distributed_point_source_line),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_line
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_line,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_y_distributed_point_source_line = csc_matrix(
                (
                    np.array(grad_shape_func_b_y_value_distributed_point_source_line),
                    (
                        np.array(phi_b_nonzero_index_row_distributed_point_source_line),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_line
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_line,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_x_times_det_J_b_time_weight_distributed_point_source_line = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_line
                    ),
                    (
                        np.array(phi_b_nonzero_index_row_distributed_point_source_line),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_line
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_line,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_y_times_det_J_b_time_weight_distributed_point_source_line = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_line
                    ),
                    (
                        np.array(phi_b_nonzero_index_row_distributed_point_source_line),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_line
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_line,
                    num_nodes_electrolyte,
                ),
            )
    if dimention == 3:
        print("yes")
        # shape_func_b_value_electrode, shape_func_b_times_det_J_b_time_weight_value_electrode, grad_shape_func_b_x_value_electrode, grad_shape_func_b_y_value_electrode,grad_shape_func_b_z_value_electrode, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode = shape_grad_shape_func(x_G_b_electrode,x_nodes_electrode, num_non_zero_phi_a_b_electrode,HT0, M_b_electrode, M_b_P_x_electrode, M_b_P_y_electrode, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrode, phi_b_P_x_nonzerovalue_data_electrode, phi_b_P_y_nonzerovalue_data_electrode, phi_b_nonzero_index_row_electrode, phi_b_nonzero_index_column_electrode, det_J_b_time_weight_electrode, IM_RKPM, M_b_P_z_electrode, HT3, phi_b_P_z_nonzerovalue_data_electrode)
        # shape_func_b_value_pore, shape_func_b_times_det_J_b_time_weight_value_pore, grad_shape_func_b_x_value_pore, grad_shape_func_b_y_value_pore,grad_shape_func_b_z_value_pore, grad_shape_func_b_x_times_det_J_b_time_weight_value_pore, grad_shape_func_b_y_times_det_J_b_time_weight_value_pore,grad_shape_func_b_z_times_det_J_b_time_weight_value_pore = shape_grad_shape_func(x_G_b_pore,x_nodes_pore, num_non_zero_phi_a_b_pore,HT0, M_b_pore, M_b_P_x_pore, M_b_P_y_pore, differential_method, HT1, HT2, phi_b_nonzerovalue_data_pore, phi_b_P_x_nonzerovalue_data_pore, phi_b_P_y_nonzerovalue_data_pore, phi_b_nonzero_index_row_pore, phi_b_nonzero_index_column_pore, det_J_b_time_weight_pore, IM_RKPM, M_b_P_z_pore, HT3, phi_b_P_z_nonzerovalue_data_pore)

        # shape_func_b_value_electrolyte_electrode_electrolyte, shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte, grad_shape_func_b_x_value_electrolyte_electrode_electrolyte, grad_shape_func_b_y_value_electrolyte_electrode_electrolyte,grad_shape_func_b_z_value_electrolyte_electrode_electrolyte, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte = shape_grad_shape_func(x_G_b_interface_electrode_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_b_electrolyte_electrode_electrolyte,HT0, M_b_electrolyte_electrode_electrolyte, M_b_P_x_electrolyte_electrode_electrolyte, M_b_P_y_electrolyte_electrode_electrolyte, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrolyte, phi_b_nonzero_index_row_electrolyte_electrode_electrolyte, phi_b_nonzero_index_column_electrolyte_electrode_electrolyte, det_J_b_time_weight_interface_electrode_electrolyte, IM_RKPM, M_b_P_z_electrolyte_electrode_electrolyte, HT3, phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrolyte)

        # shape_func_b_value_electrolyte_electrode_electrode, shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode, grad_shape_func_b_x_value_electrolyte_electrode_electrode, grad_shape_func_b_y_value_electrolyte_electrode_electrode,grad_shape_func_b_z_value_electrolyte_electrode_electrode, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode = shape_grad_shape_func(x_G_b_interface_electrode_electrolyte,x_nodes_electrode, num_non_zero_phi_a_b_electrolyte_electrode_electrode,HT0, M_b_electrolyte_electrode_electrode, M_b_P_x_electrolyte_electrode_electrode, M_b_P_y_electrolyte_electrode_electrode, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_x_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_P_y_nonzerovalue_data_electrolyte_electrode_electrode, phi_b_nonzero_index_row_electrolyte_electrode_electrode, phi_b_nonzero_index_column_electrolyte_electrode_electrode, det_J_b_time_weight_interface_electrode_electrolyte, IM_RKPM, M_b_P_z_electrolyte_electrode_electrode, HT3, phi_b_P_z_nonzerovalue_data_electrolyte_electrode_electrode)

        # shape_func_b_value_electrode_pore_electrode, shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode, grad_shape_func_b_x_value_electrode_pore_electrode, grad_shape_func_b_y_value_electrode_pore_electrode,grad_shape_func_b_z_value_electrode_pore_electrode, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode = shape_grad_shape_func(x_G_b_interface_electrode_pore,x_nodes_electrode, num_non_zero_phi_a_b_electrode_pore_electrode,HT0, M_b_electrode_pore_electrode, M_b_P_x_electrode_pore_electrode, M_b_P_y_electrode_pore_electrode, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrode_pore_electrode, phi_b_P_x_nonzerovalue_data_electrode_pore_electrode, phi_b_P_y_nonzerovalue_data_electrode_pore_electrode, phi_b_nonzero_index_row_electrode_pore_electrode, phi_b_nonzero_index_column_electrode_pore_electrode, det_J_b_time_weight_interface_electrode_pore, IM_RKPM, M_b_P_z_electrode_pore_electrode, HT3, phi_b_P_z_nonzerovalue_data_electrode_pore_electrode)

        # shape_func_b_value_electrode_pore_pore, shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore, grad_shape_func_b_x_value_electrode_pore_pore, grad_shape_func_b_y_value_electrode_pore_pore,grad_shape_func_b_z_value_electrode_pore_pore, grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore, grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore,grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore = shape_grad_shape_func(x_G_b_interface_electrode_pore,x_nodes_pore, num_non_zero_phi_a_b_electrode_pore_pore,HT0, M_b_electrode_pore_pore, M_b_P_x_electrode_pore_pore, M_b_P_y_electrode_pore_pore, differential_method, HT1, HT2, phi_b_nonzerovalue_data_electrode_pore_pore, phi_b_P_x_nonzerovalue_data_electrode_pore_pore, phi_b_P_y_nonzerovalue_data_electrode_pore_pore, phi_b_nonzero_index_row_electrode_pore_pore, phi_b_nonzero_index_column_electrode_pore_pore, det_J_b_time_weight_interface_electrode_pore, IM_RKPM, M_b_P_z_electrode_pore_pore, HT3, phi_b_P_z_nonzerovalue_data_electrode_pore_pore)

        if delta_point_source == "False":
            num_gauss_points_distributed_point_source_surface = np.shape(
                x_G_b_distributed_point_source_surface
            )[0]

            # shape_func_b_value_distributed_point_source_surface, shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface, grad_shape_func_b_x_value_distributed_point_source_surface, grad_shape_func_b_y_value_distributed_point_source_surface,grad_shape_func_b_z_value_distributed_point_source_surface, grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface, grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface,grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface = shape_grad_shape_func(x_G_b_distributed_point_source_surface,x_nodes_electrolyte, num_non_zero_phi_a_b_distributed_point_source_surface,HT0, M_b_distributed_point_source_surface, M_b_P_x_distributed_point_source_surface, M_b_P_y_distributed_point_source_surface, differential_method, HT1, HT2, phi_b_nonzerovalue_data_distributed_point_source_surface, phi_b_P_x_nonzerovalue_data_distributed_point_source_surface, phi_b_P_y_nonzerovalue_data_distributed_point_source_surface, phi_b_nonzero_index_row_distributed_point_source_surface, phi_b_nonzero_index_column_distributed_point_source_surface, det_J_b_time_weight_distributed_point_source_surface, IM_RKPM, M_b_P_z_distributed_point_source_surface, HT3, phi_b_P_z_nonzerovalue_data_distributed_point_source_surface)

            # np.savetxt('shape_func_b_value_distributed_point_source_surface.txt', np.asarray(shape_func_b_value_distributed_point_source_surface))
            # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface))
            # np.savetxt('grad_shape_func_b_x_value_distributed_point_source_surface.txt', np.asarray(grad_shape_func_b_x_value_distributed_point_source_surface))
            # np.savetxt('grad_shape_func_b_y_value_distributed_point_source_surface.txt', np.asarray(grad_shape_func_b_y_value_distributed_point_source_surface))
            # np.savetxt('grad_shape_func_b_z_value_distributed_point_source_surface.txt', np.asarray(grad_shape_func_b_z_value_distributed_point_source_surface))
            # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface))
            # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface))
            # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface))
            shape_func_b_value_distributed_point_source_surface = np.loadtxt(
                "shape_func_b_value_distributed_point_source_surface.txt"
            )
            shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface = np.loadtxt(
                "shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface.txt"
            )
            grad_shape_func_b_x_value_distributed_point_source_surface = np.loadtxt(
                "grad_shape_func_b_x_value_distributed_point_source_surface.txt"
            )
            grad_shape_func_b_y_value_distributed_point_source_surface = np.loadtxt(
                "grad_shape_func_b_y_value_distributed_point_source_surface.txt"
            )
            grad_shape_func_b_z_value_distributed_point_source_surface = np.loadtxt(
                "grad_shape_func_b_z_value_distributed_point_source_surface.txt"
            )
            grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface = np.loadtxt(
                "grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface.txt"
            )
            grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface = np.loadtxt(
                "grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface.txt"
            )
            grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface = np.loadtxt(
                "grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface.txt"
            )

            shape_func_b_distributed_point_source_surface = csc_matrix(
                (
                    np.array(shape_func_b_value_distributed_point_source_surface),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            shape_func_b_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        shape_func_b_times_det_J_b_time_weight_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_x_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_x_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_y_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_y_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_z_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_z_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_x_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_x_times_det_J_b_time_weight_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_y_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_y_times_det_J_b_time_weight_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )
            grad_shape_func_b_z_times_det_J_b_time_weight_distributed_point_source_surface = csc_matrix(
                (
                    np.array(
                        grad_shape_func_b_z_times_det_J_b_time_weight_value_distributed_point_source_surface
                    ),
                    (
                        np.array(
                            phi_b_nonzero_index_row_distributed_point_source_surface
                        ),
                        np.array(
                            phi_b_nonzero_index_column_distributed_point_source_surface
                        ),
                    ),
                ),
                shape=(
                    num_gauss_points_distributed_point_source_surface,
                    num_nodes_electrolyte,
                ),
            )

    # np.savetxt('shape_func_b_value_electrolyte_electrode_electrolyte.txt', np.asarray(shape_func_b_value_electrolyte_electrode_electrolyte))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte))
    # np.savetxt('grad_shape_func_b_x_value_electrolyte_electrode_electrolyte.txt', np.asarray(grad_shape_func_b_x_value_electrolyte_electrode_electrolyte))
    # np.savetxt('grad_shape_func_b_y_value_electrolyte_electrode_electrolyte.txt', np.asarray(grad_shape_func_b_y_value_electrolyte_electrode_electrolyte))
    # np.savetxt('grad_shape_func_b_z_value_electrolyte_electrode_electrolyte.txt', np.asarray(grad_shape_func_b_z_value_electrolyte_electrode_electrolyte))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte))
    shape_func_b_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "shape_func_b_value_electrolyte_electrode_electrolyte.txt"
    )
    shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt"
    )
    grad_shape_func_b_x_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "grad_shape_func_b_x_value_electrolyte_electrode_electrolyte.txt"
    )
    grad_shape_func_b_y_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "grad_shape_func_b_y_value_electrolyte_electrode_electrolyte.txt"
    )
    grad_shape_func_b_z_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "grad_shape_func_b_z_value_electrolyte_electrode_electrolyte.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte.txt"
    )

    # np.savetxt('shape_func_b_value_electrolyte_electrode_electrode.txt', np.asarray(shape_func_b_value_electrolyte_electrode_electrode))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode))
    # np.savetxt('grad_shape_func_b_x_value_electrolyte_electrode_electrode.txt', np.asarray(grad_shape_func_b_x_value_electrolyte_electrode_electrode))
    # np.savetxt('grad_shape_func_b_y_value_electrolyte_electrode_electrode.txt', np.asarray(grad_shape_func_b_y_value_electrolyte_electrode_electrode))
    # np.savetxt('grad_shape_func_b_z_value_electrolyte_electrode_electrode.txt', np.asarray(grad_shape_func_b_z_value_electrolyte_electrode_electrode))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode))
    shape_func_b_value_electrolyte_electrode_electrode = np.loadtxt(
        "shape_func_b_value_electrolyte_electrode_electrode.txt"
    )
    shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt"
    )
    grad_shape_func_b_x_value_electrolyte_electrode_electrode = np.loadtxt(
        "grad_shape_func_b_x_value_electrolyte_electrode_electrode.txt"
    )
    grad_shape_func_b_y_value_electrolyte_electrode_electrode = np.loadtxt(
        "grad_shape_func_b_y_value_electrolyte_electrode_electrode.txt"
    )
    grad_shape_func_b_z_value_electrolyte_electrode_electrode = np.loadtxt(
        "grad_shape_func_b_z_value_electrolyte_electrode_electrode.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode.txt"
    )

    # np.savetxt('shape_func_b_value_electrode_pore_electrode.txt', np.asarray(shape_func_b_value_electrode_pore_electrode))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode))
    # np.savetxt('grad_shape_func_b_x_value_electrode_pore_electrode.txt', np.asarray(grad_shape_func_b_x_value_electrode_pore_electrode))
    # np.savetxt('grad_shape_func_b_y_value_electrode_pore_electrode.txt', np.asarray(grad_shape_func_b_y_value_electrode_pore_electrode))
    # np.savetxt('grad_shape_func_b_z_value_electrode_pore_electrode.txt', np.asarray(grad_shape_func_b_z_value_electrode_pore_electrode))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode))
    shape_func_b_value_electrode_pore_electrode = np.loadtxt(
        "shape_func_b_value_electrode_pore_electrode.txt"
    )
    shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode.txt"
    )
    grad_shape_func_b_x_value_electrode_pore_electrode = np.loadtxt(
        "grad_shape_func_b_x_value_electrode_pore_electrode.txt"
    )
    grad_shape_func_b_y_value_electrode_pore_electrode = np.loadtxt(
        "grad_shape_func_b_y_value_electrode_pore_electrode.txt"
    )
    grad_shape_func_b_z_value_electrode_pore_electrode = np.loadtxt(
        "grad_shape_func_b_z_value_electrode_pore_electrode.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode.txt"
    )

    # np.savetxt('shape_func_b_value_electrode_pore_pore.txt', np.asarray(shape_func_b_value_electrode_pore_pore))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore))
    # np.savetxt('grad_shape_func_b_x_value_electrode_pore_pore.txt', np.asarray(grad_shape_func_b_x_value_electrode_pore_pore))
    # np.savetxt('grad_shape_func_b_y_value_electrode_pore_pore.txt', np.asarray(grad_shape_func_b_y_value_electrode_pore_pore))
    # np.savetxt('grad_shape_func_b_z_value_electrode_pore_pore.txt', np.asarray(grad_shape_func_b_z_value_electrode_pore_pore))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore))
    shape_func_b_value_electrode_pore_pore = np.loadtxt(
        "shape_func_b_value_electrode_pore_pore.txt"
    )
    shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore.txt"
    )
    grad_shape_func_b_x_value_electrode_pore_pore = np.loadtxt(
        "grad_shape_func_b_x_value_electrode_pore_pore.txt"
    )
    grad_shape_func_b_y_value_electrode_pore_pore = np.loadtxt(
        "grad_shape_func_b_y_value_electrode_pore_pore.txt"
    )
    grad_shape_func_b_z_value_electrode_pore_pore = np.loadtxt(
        "grad_shape_func_b_z_value_electrode_pore_pore.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_pore.txt"
    )

    # np.savetxt('shape_func_b_value_electrode.txt', np.asarray(shape_func_b_value_electrode))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_electrode.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_electrode))
    # np.savetxt('grad_shape_func_b_x_value_electrode.txt', np.asarray(grad_shape_func_b_x_value_electrode))
    # np.savetxt('grad_shape_func_b_y_value_electrode.txt', np.asarray(grad_shape_func_b_y_value_electrode))
    # np.savetxt('grad_shape_func_b_z_value_electrode.txt', np.asarray(grad_shape_func_b_z_value_electrode))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode))
    shape_func_b_value_electrode = np.loadtxt("shape_func_b_value_electrode.txt")
    shape_func_b_times_det_J_b_time_weight_value_electrode = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_electrode.txt"
    )
    grad_shape_func_b_x_value_electrode = np.loadtxt(
        "grad_shape_func_b_x_value_electrode.txt"
    )
    grad_shape_func_b_y_value_electrode = np.loadtxt(
        "grad_shape_func_b_y_value_electrode.txt"
    )
    grad_shape_func_b_z_value_electrode = np.loadtxt(
        "grad_shape_func_b_z_value_electrode.txt"
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode.txt"
    )

    # np.savetxt('shape_func_b_value_pore.txt', np.asarray(shape_func_b_value_pore))
    # np.savetxt('shape_func_b_times_det_J_b_time_weight_value_pore.txt', np.asarray(shape_func_b_times_det_J_b_time_weight_value_pore))
    # np.savetxt('grad_shape_func_b_x_value_pore.txt', np.asarray(grad_shape_func_b_x_value_pore))
    # np.savetxt('grad_shape_func_b_y_value_pore.txt', np.asarray(grad_shape_func_b_y_value_pore))
    # np.savetxt('grad_shape_func_b_z_value_pore.txt', np.asarray(grad_shape_func_b_z_value_pore))
    # np.savetxt('grad_shape_func_b_x_times_det_J_b_time_weight_value_pore.txt', np.asarray(grad_shape_func_b_x_times_det_J_b_time_weight_value_pore))
    # np.savetxt('grad_shape_func_b_y_times_det_J_b_time_weight_value_pore.txt', np.asarray(grad_shape_func_b_y_times_det_J_b_time_weight_value_pore))
    # np.savetxt('grad_shape_func_b_z_times_det_J_b_time_weight_value_pore.txt', np.asarray(grad_shape_func_b_z_times_det_J_b_time_weight_value_pore))
    shape_func_b_value_pore = np.loadtxt("shape_func_b_value_pore.txt")
    shape_func_b_times_det_J_b_time_weight_value_pore = np.loadtxt(
        "shape_func_b_times_det_J_b_time_weight_value_pore.txt"
    )
    grad_shape_func_b_x_value_pore = np.loadtxt("grad_shape_func_b_x_value_pore.txt")
    grad_shape_func_b_y_value_pore = np.loadtxt("grad_shape_func_b_y_value_pore.txt")
    grad_shape_func_b_z_value_pore = np.loadtxt("grad_shape_func_b_z_value_pore.txt")
    grad_shape_func_b_x_times_det_J_b_time_weight_value_pore = np.loadtxt(
        "grad_shape_func_b_x_times_det_J_b_time_weight_value_pore.txt"
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_value_pore = np.loadtxt(
        "grad_shape_func_b_y_times_det_J_b_time_weight_value_pore.txt"
    )
    grad_shape_func_b_z_times_det_J_b_time_weight_value_pore = np.loadtxt(
        "grad_shape_func_b_z_times_det_J_b_time_weight_value_pore.txt"
    )

    shape_func_b_electrolyte_electrode_electrolyte = csc_matrix(
        (
            np.array(shape_func_b_value_electrolyte_electrode_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrolyte,
        ),
    )
    shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix(
        (
            np.array(
                shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte
            ),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrolyte,
        ),
    )
    grad_shape_func_b_x_electrolyte_electrode_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_electrolyte_electrode_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrolyte,
        ),
    )
    grad_shape_func_b_y_electrolyte_electrode_electrolyte = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_electrolyte_electrode_electrolyte),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrolyte,
        ),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix(
        (
            np.array(
                grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte
            ),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrolyte,
        ),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix(
        (
            np.array(
                grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte
            ),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrolyte),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrolyte,
        ),
    )

    shape_func_b_electrolyte_electrode_electrode = csc_matrix(
        (
            np.array(shape_func_b_value_electrolyte_electrode_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrode,
        ),
    )
    shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix(
        (
            np.array(
                shape_func_b_times_det_J_b_time_weight_value_electrolyte_electrode_electrode
            ),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrode,
        ),
    )
    grad_shape_func_b_x_electrolyte_electrode_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_electrolyte_electrode_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrode,
        ),
    )
    grad_shape_func_b_y_electrolyte_electrode_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_electrolyte_electrode_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrode,
        ),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix(
        (
            np.array(
                grad_shape_func_b_x_times_det_J_b_time_weight_value_electrolyte_electrode_electrode
            ),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrode,
        ),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix(
        (
            np.array(
                grad_shape_func_b_y_times_det_J_b_time_weight_value_electrolyte_electrode_electrode
            ),
            (
                np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                np.array(phi_b_nonzero_index_column_electrolyte_electrode_electrode),
            ),
        ),
        shape=(
            num_gauss_points_on_electrolyte_electrode_interface,
            num_nodes_electrode,
        ),
    )

    shape_func_b_electrode_pore_electrode = csc_matrix(
        (
            np.array(shape_func_b_value_electrode_pore_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
    )
    shape_func_b_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix(
        (
            np.array(
                shape_func_b_times_det_J_b_time_weight_value_electrode_pore_electrode
            ),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
    )
    grad_shape_func_b_x_electrode_pore_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_electrode_pore_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
    )
    grad_shape_func_b_y_electrode_pore_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_electrode_pore_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix(
        (
            np.array(
                grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_electrode
            ),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix(
        (
            np.array(
                grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_electrode
            ),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
    )

    shape_func_b_electrode_pore_pore = csc_matrix(
        (
            np.array(shape_func_b_value_electrode_pore_pore),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                np.array(phi_b_nonzero_index_column_electrode_pore_pore),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
    )
    shape_func_b_times_det_J_b_time_weight_electrode_pore_pore = csc_matrix(
        (
            np.array(shape_func_b_times_det_J_b_time_weight_value_electrode_pore_pore),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                np.array(phi_b_nonzero_index_column_electrode_pore_pore),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
    )
    grad_shape_func_b_x_electrode_pore_pore = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_electrode_pore_pore),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                np.array(phi_b_nonzero_index_column_electrode_pore_pore),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
    )
    grad_shape_func_b_y_electrode_pore_pore = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_electrode_pore_pore),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                np.array(phi_b_nonzero_index_column_electrode_pore_pore),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_electrode_pore_pore = csc_matrix(
        (
            np.array(
                grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode_pore_pore
            ),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                np.array(phi_b_nonzero_index_column_electrode_pore_pore),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_electrode_pore_pore = csc_matrix(
        (
            np.array(
                grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode_pore_pore
            ),
            (
                np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                np.array(phi_b_nonzero_index_column_electrode_pore_pore),
            ),
        ),
        shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
    )

    shape_func_b_electrode = csc_matrix(
        (
            np.array(shape_func_b_value_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode),
                np.array(phi_b_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
    )
    shape_func_b_times_det_J_b_time_weight_electrode = csc_matrix(
        (
            np.array(shape_func_b_times_det_J_b_time_weight_value_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode),
                np.array(phi_b_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
    )
    grad_shape_func_b_x_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode),
                np.array(phi_b_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
    )
    grad_shape_func_b_y_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode),
                np.array(phi_b_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode),
                np.array(phi_b_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_electrode = csc_matrix(
        (
            np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_electrode),
            (
                np.array(phi_b_nonzero_index_row_electrode),
                np.array(phi_b_nonzero_index_column_electrode),
            ),
        ),
        shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
    )

    shape_func_b_pore = csc_matrix(
        (
            np.array(shape_func_b_value_pore),
            (
                np.array(phi_b_nonzero_index_row_pore),
                np.array(phi_b_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
    )
    shape_func_b_times_det_J_b_time_weight_pore = csc_matrix(
        (
            np.array(shape_func_b_times_det_J_b_time_weight_value_pore),
            (
                np.array(phi_b_nonzero_index_row_pore),
                np.array(phi_b_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
    )
    grad_shape_func_b_x_pore = csc_matrix(
        (
            np.array(grad_shape_func_b_x_value_pore),
            (
                np.array(phi_b_nonzero_index_row_pore),
                np.array(phi_b_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
    )
    grad_shape_func_b_y_pore = csc_matrix(
        (
            np.array(grad_shape_func_b_y_value_pore),
            (
                np.array(phi_b_nonzero_index_row_pore),
                np.array(phi_b_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
    )
    grad_shape_func_b_x_times_det_J_b_time_weight_pore = csc_matrix(
        (
            np.array(grad_shape_func_b_x_times_det_J_b_time_weight_value_pore),
            (
                np.array(phi_b_nonzero_index_row_pore),
                np.array(phi_b_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
    )
    grad_shape_func_b_y_times_det_J_b_time_weight_pore = csc_matrix(
        (
            np.array(grad_shape_func_b_y_times_det_J_b_time_weight_value_pore),
            (
                np.array(phi_b_nonzero_index_row_pore),
                np.array(phi_b_nonzero_index_column_pore),
            ),
        ),
        shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
    )

    if dimention == 3:
        grad_shape_func_b_z_electrode = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_electrode),
                (
                    np.array(phi_b_nonzero_index_row_electrode),
                    np.array(phi_b_nonzero_index_column_electrode),
                ),
            ),
            shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_electrode = csc_matrix(
            (
                np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode),
                (
                    np.array(phi_b_nonzero_index_row_electrode),
                    np.array(phi_b_nonzero_index_column_electrode),
                ),
            ),
            shape=(num_gauss_points_on_boundary_electrode, num_nodes_electrode),
        )

        grad_shape_func_b_z_pore = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_pore),
                (
                    np.array(phi_b_nonzero_index_row_pore),
                    np.array(phi_b_nonzero_index_column_pore),
                ),
            ),
            shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_pore = csc_matrix(
            (
                np.array(grad_shape_func_b_z_times_det_J_b_time_weight_value_pore),
                (
                    np.array(phi_b_nonzero_index_row_pore),
                    np.array(phi_b_nonzero_index_column_pore),
                ),
            ),
            shape=(num_gauss_points_on_boundary_pore, num_nodes_pore),
        )

        grad_shape_func_b_z_electrolyte_electrode_electrolyte = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_electrolyte_electrode_electrolyte),
                (
                    np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                    np.array(
                        phi_b_nonzero_index_column_electrolyte_electrode_electrolyte
                    ),
                ),
            ),
            shape=(
                num_gauss_points_on_electrolyte_electrode_interface,
                num_nodes_electrolyte,
            ),
        )
        grad_shape_func_b_z_electrolyte_electrode_electrode = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_electrolyte_electrode_electrode),
                (
                    np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                    np.array(
                        phi_b_nonzero_index_column_electrolyte_electrode_electrode
                    ),
                ),
            ),
            shape=(
                num_gauss_points_on_electrolyte_electrode_interface,
                num_nodes_electrode,
            ),
        )
        grad_shape_func_b_z_electrode_pore_electrode = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_electrode_pore_electrode),
                (
                    np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                    np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
                ),
            ),
            shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
        )
        grad_shape_func_b_z_electrode_pore_pore = csc_matrix(
            (
                np.array(grad_shape_func_b_z_value_electrode_pore_pore),
                (
                    np.array(phi_b_nonzero_index_row_electrode_pore_pore),
                    np.array(phi_b_nonzero_index_column_electrode_pore_pore),
                ),
            ),
            shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_pore),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte_electrode_electrolyte = csc_matrix(
            (
                np.array(
                    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrolyte
                ),
                (
                    np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrolyte),
                    np.array(
                        phi_b_nonzero_index_column_electrolyte_electrode_electrolyte
                    ),
                ),
            ),
            shape=(
                num_gauss_points_on_electrolyte_electrode_interface,
                num_nodes_electrolyte,
            ),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte_electrode_electrode = csc_matrix(
            (
                np.array(
                    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrolyte_electrode_electrode
                ),
                (
                    np.array(phi_b_nonzero_index_row_electrolyte_electrode_electrode),
                    np.array(
                        phi_b_nonzero_index_column_electrolyte_electrode_electrode
                    ),
                ),
            ),
            shape=(
                num_gauss_points_on_electrolyte_electrode_interface,
                num_nodes_electrode,
            ),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix(
            (
                np.array(
                    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode
                ),
                (
                    np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                    np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
                ),
            ),
            shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
        )
        grad_shape_func_b_z_times_det_J_b_time_weight_electrode_pore_electrode = csc_matrix(
            (
                np.array(
                    grad_shape_func_b_z_times_det_J_b_time_weight_value_electrode_pore_electrode
                ),
                (
                    np.array(phi_b_nonzero_index_row_electrode_pore_electrode),
                    np.array(phi_b_nonzero_index_column_electrode_pore_electrode),
                ),
            ),
            shape=(num_gauss_points_on_electrode_pore_interface, num_nodes_electrode),
        )

    """shape function with size n_nodes times n_nodes, this is used to predict the potential on all nodes"""

    if dimention == 2:
        M_electrolyte_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_electrolyte)]
        )
        M_P_x_electrolyte_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_electrolyte)]
        )  # partial M partial x
        M_P_y_electrolyte_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_electrolyte)]
        )  # partial M partial y
        (
            phi_nonzero_index_row_electrolyte_nn,
            phi_nonzero_index_column_electrolyte_nn,
            phi_nonzerovalue_data_electrolyte_nn,
            phi_P_x_nonzerovalue_data_electrolyte_nn,
            phi_P_y_nonzerovalue_data_electrolyte_nn,
            phi_P_z_nonzerovalue_data_electrolyte_nn,
            M_electrolyte_nn,
            M_P_x_electrolyte_nn,
            M_P_y_electrolyte_nn,
            M_P_z_electrolyte_nn,
        ) = compute_phi_M(
            x_nodes_electrolyte,
            Gauss_grain_id_electrolyte,
            x_nodes_electrolyte,
            nodes_grain_id_electrolyte,
            a_electrolyte,
            M_electrolyte_nn,
            M_P_x_electrolyte_nn,
            M_P_y_electrolyte_nn,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_electrode_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_electrode)]
        )
        M_P_x_electrode_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_electrode)]
        )  # partial M partial x
        M_P_y_electrode_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_electrode)]
        )  # partial M partial y
        (
            phi_nonzero_index_row_electrode_nn,
            phi_nonzero_index_column_electrode_nn,
            phi_nonzerovalue_data_electrode_nn,
            phi_P_x_nonzerovalue_data_electrode_nn,
            phi_P_y_nonzerovalue_data_electrode_nn,
            phi_P_z_nonzerovalue_data_electrode_nn,
            M_electrode_nn,
            M_P_x_electrode_nn,
            M_P_y_electrode_nn,
            M_P_z_electrode_nn,
        ) = compute_phi_M(
            x_nodes_electrode,
            Gauss_grain_id_electrode,
            x_nodes_electrode,
            nodes_grain_id_electrode,
            a_electrode,
            M_electrode_nn,
            M_P_x_electrode_nn,
            M_P_y_electrode_nn,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

        M_pore_nn = np.array([np.zeros((3, 3)) for _ in range(num_nodes_pore)])
        M_P_x_pore_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_pore)]
        )  # partial M partial x
        M_P_y_pore_nn = np.array(
            [np.zeros((3, 3)) for _ in range(num_nodes_pore)]
        )  # partial M partial y
        (
            phi_nonzero_index_row_pore_nn,
            phi_nonzero_index_column_pore_nn,
            phi_nonzerovalue_data_pore_nn,
            phi_P_x_nonzerovalue_data_pore_nn,
            phi_P_y_nonzerovalue_data_pore_nn,
            phi_P_z_nonzerovalue_data_pore_nn,
            M_pore_nn,
            M_P_x_pore_nn,
            M_P_y_pore_nn,
            M_P_z_pore_nn,
        ) = compute_phi_M(
            x_nodes_pore,
            Gauss_grain_id_pore,
            x_nodes_pore,
            nodes_grain_id_pore,
            a_pore,
            M_pore_nn,
            M_P_x_pore_nn,
            M_P_y_pore_nn,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )

    if dimention == 3:
        M_electrolyte_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrolyte)]
        )
        M_P_x_electrolyte_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrolyte)]
        )  # partial M partial x
        M_P_y_electrolyte_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrolyte)]
        )  # partial M partial y
        M_P_z_electrolyte_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrolyte)]
        )  # partial M partial y
        # phi_nonzero_index_row_electrolyte_nn, phi_nonzero_index_column_electrolyte_nn, phi_nonzerovalue_data_electrolyte_nn, phi_P_x_nonzerovalue_data_electrolyte_nn, phi_P_y_nonzerovalue_data_electrolyte_nn, phi_P_z_nonzerovalue_data_electrolyte_nn, M_electrolyte_nn, M_P_x_electrolyte_nn, M_P_y_electrolyte_nn, M_P_z_electrolyte_nn = compute_phi_M(x_nodes_electrolyte, Gauss_grain_id_electrolyte, x_nodes_electrolyte,nodes_grain_id_electrolyte, a_electrolyte, M_electrolyte_nn, M_P_x_electrolyte_nn, M_P_y_electrolyte_nn, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrolyte_nn)

        M_electrode_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrode)]
        )
        M_P_x_electrode_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrode)]
        )  # partial M partial x
        M_P_y_electrode_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrode)]
        )  # partial M partial y
        M_P_z_electrode_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_electrode)]
        )  # partial M partial y
        # phi_nonzero_index_row_electrode_nn, phi_nonzero_index_column_electrode_nn, phi_nonzerovalue_data_electrode_nn, phi_P_x_nonzerovalue_data_electrode_nn, phi_P_y_nonzerovalue_data_electrode_nn, phi_P_z_nonzerovalue_data_electrode_nn, M_electrode_nn, M_P_x_electrode_nn, M_P_y_electrode_nn, M_P_z_electrode_nn = compute_phi_M(x_nodes_electrode, Gauss_grain_id_electrode, x_nodes_electrode,nodes_grain_id_electrode, a_electrode, M_electrode_nn, M_P_x_electrode_nn, M_P_y_electrode_nn, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrode_nn)

        M_pore_nn = np.array([np.zeros((4, 4)) for _ in range(num_nodes_pore)])
        M_P_x_pore_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_pore)]
        )  # partial M partial x
        M_P_y_pore_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_pore)]
        )  # partial M partial y
        M_P_z_pore_nn = np.array(
            [np.zeros((4, 4)) for _ in range(num_nodes_pore)]
        )  # partial M partial y
        # phi_nonzero_index_row_pore_nn, phi_nonzero_index_column_pore_nn, phi_nonzerovalue_data_pore_nn, phi_P_x_nonzerovalue_data_pore_nn, phi_P_y_nonzerovalue_data_pore_nn, phi_P_z_nonzerovalue_data_pore_nn, M_pore_nn, M_P_x_pore_nn, M_P_y_pore_nn, M_P_z_pore_nn = compute_phi_M(x_nodes_pore, Gauss_grain_id_pore, x_nodes_pore,nodes_grain_id_pore, a_pore, M_pore_nn, M_P_x_pore_nn, M_P_y_pore_nn, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_pore_nn)

    # np.savetxt('phi_nonzero_index_row_electrode_nn.txt', phi_nonzero_index_row_electrode_nn)
    # np.savetxt('phi_nonzero_index_column_electrode_nn.txt', phi_nonzero_index_column_electrode_nn)
    phi_nonzero_index_row_electrode_nn = np.loadtxt(
        "phi_nonzero_index_row_electrode_nn.txt"
    )
    phi_nonzero_index_column_electrode_nn = np.loadtxt(
        "phi_nonzero_index_column_electrode_nn.txt"
    )

    # np.savetxt('phi_nonzero_index_row_electrolyte_nn.txt', phi_nonzero_index_row_electrolyte_nn)
    # np.savetxt('phi_nonzero_index_column_electrolyte_nn.txt', phi_nonzero_index_column_electrolyte_nn)
    phi_nonzero_index_row_electrolyte_nn = np.loadtxt(
        "phi_nonzero_index_row_electrolyte_nn.txt"
    )
    phi_nonzero_index_column_electrolyte_nn = np.loadtxt(
        "phi_nonzero_index_column_electrolyte_nn.txt"
    )

    # np.savetxt('phi_nonzero_index_row_pore_nn.txt', phi_nonzero_index_row_pore_nn)
    # np.savetxt('phi_nonzero_index_column_pore_nn.txt', phi_nonzero_index_column_pore_nn)
    phi_nonzero_index_row_pore_nn = np.loadtxt("phi_nonzero_index_row_pore_nn.txt")
    phi_nonzero_index_column_pore_nn = np.loadtxt(
        "phi_nonzero_index_column_pore_nn.txt"
    )

    num_non_zero_phi_a_electrolyte_nn = np.shape(
        np.array(phi_nonzero_index_row_electrolyte_nn)
    )[0]
    # shape_func_value_electrolyte_nn = shape_func_n_nodes_by_n_nodes(x_nodes_electrolyte,x_nodes_electrolyte, num_non_zero_phi_a_electrolyte_nn,HT0, M_electrolyte_nn, phi_nonzerovalue_data_electrolyte_nn,phi_nonzero_index_row_electrolyte_nn, phi_nonzero_index_column_electrolyte_nn)
    # np.savetxt('shape_func_value_electrolyte_nn.txt', np.asarray(shape_func_value_electrolyte_nn))
    shape_func_value_electrolyte_nn = np.loadtxt("shape_func_value_electrolyte_nn.txt")
    shape_func_n_nodes_n_nodes_electrolyte = csc_matrix(
        (
            np.array(shape_func_value_electrolyte_nn),
            (
                np.array(phi_nonzero_index_row_electrolyte_nn),
                np.array(phi_nonzero_index_column_electrolyte_nn),
            ),
        ),
        shape=(num_nodes_electrolyte, num_nodes_electrolyte),
    )

    num_non_zero_phi_a_electrode_nn = np.shape(
        np.array(phi_nonzero_index_row_electrode_nn)
    )[0]
    # shape_func_value_electrode_nn = shape_func_n_nodes_by_n_nodes(x_nodes_electrode,x_nodes_electrode, num_non_zero_phi_a_electrode_nn,HT0, M_electrode_nn, phi_nonzerovalue_data_electrode_nn,phi_nonzero_index_row_electrode_nn, phi_nonzero_index_column_electrode_nn)
    # np.savetxt('shape_func_value_electrode_nn.txt', np.asarray(shape_func_value_electrode_nn))
    shape_func_value_electrode_nn = np.loadtxt("shape_func_value_electrode_nn.txt")
    shape_func_n_nodes_n_nodes_electrode = csc_matrix(
        (
            np.array(shape_func_value_electrode_nn),
            (
                np.array(phi_nonzero_index_row_electrode_nn),
                np.array(phi_nonzero_index_column_electrode_nn),
            ),
        ),
        shape=(num_nodes_electrode, num_nodes_electrode),
    )

    num_non_zero_phi_a_pore_nn = np.shape(np.array(phi_nonzero_index_row_pore_nn))[0]
    # shape_func_value_pore_nn = shape_func_n_nodes_by_n_nodes(x_nodes_pore,x_nodes_pore, num_non_zero_phi_a_pore_nn,HT0, M_pore_nn, phi_nonzerovalue_data_pore_nn,phi_nonzero_index_row_pore_nn, phi_nonzero_index_column_pore_nn)
    # np.savetxt('shape_func_value_pore_nn.txt', np.asarray(shape_func_value_pore_nn))
    shape_func_value_pore_nn = np.loadtxt("shape_func_value_pore_nn.txt")
    shape_func_n_nodes_n_nodes_pore = csc_matrix(
        (
            np.array(shape_func_value_pore_nn),
            (
                np.array(phi_nonzero_index_row_pore_nn),
                np.array(phi_nonzero_index_column_pore_nn),
            ),
        ),
        shape=(num_nodes_pore, num_nodes_pore),
    )

    """
    1. shape function used to interpolate the phi and phie at the aource point (2d), shape: number of source points times number of nodes
    2. shape function used to interpolate the displacmeent at the fixed point (2d), shape: number of fixed points times number of nodes
    """

    if dimention == 2:
        M_electrolyte_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )
        M_P_x_electrolyte_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )  # partial M partial x
        M_P_y_electrolyte_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )  # partial M partial y
        (
            phi_nonzero_index_row_electrolyte_source_nodes,
            phi_nonzero_index_column_electrolyte_source_nodes,
            phi_nonzerovalue_data_electrolyte_source_nodes,
            phi_P_x_nonzerovalue_data_electrolyte_source_nodes,
            phi_P_y_nonzerovalue_data_electrolyte_source_nodes,
            phi_P_z_nonzerovalue_data_electrolyte_source_nodes,
            M_electrolyte_source_nodes,
            M_P_x_electrolyte_source_nodes,
            M_P_y_electrolyte_source_nodes,
            M_P_z_electrolyte_source_nodes,
        ) = compute_phi_M(
            point_source,
            Gauss_grain_id_electrolyte,
            x_nodes_electrolyte,
            nodes_grain_id_electrolyte,
            a_electrolyte,
            M_electrolyte_source_nodes,
            M_P_x_electrolyte_source_nodes,
            M_P_y_electrolyte_source_nodes,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        np.savetxt(
            "phi_nonzero_index_row_electrolyte_source_nodes.txt",
            phi_nonzero_index_row_electrolyte_source_nodes,
        )
        np.savetxt(
            "phi_nonzero_index_column_electrolyte_source_nodes.txt",
            phi_nonzero_index_column_electrolyte_source_nodes,
        )
        phi_nonzero_index_row_electrolyte_source_nodes = np.loadtxt(
            "phi_nonzero_index_row_electrolyte_source_nodes.txt"
        )
        phi_nonzero_index_column_electrolyte_source_nodes = np.loadtxt(
            "phi_nonzero_index_column_electrolyte_source_nodes.txt"
        )
        num_non_zero_phi_a_electrolyte_source_nodes = np.shape(
            np.array(phi_nonzero_index_row_electrolyte_source_nodes)
        )[0]
        shape_func_value_electrolyte_source_nodes = shape_func_n_nodes_by_n_nodes(
            point_source,
            x_nodes_electrolyte,
            num_non_zero_phi_a_electrolyte_source_nodes,
            HT0,
            M_electrolyte_source_nodes,
            phi_nonzerovalue_data_electrolyte_source_nodes,
            phi_nonzero_index_row_electrolyte_source_nodes,
            phi_nonzero_index_column_electrolyte_source_nodes,
        )
        np.savetxt(
            "shape_func_value_electrolyte_source_nodes.txt",
            shape_func_value_electrolyte_source_nodes,
        )
        shape_func_value_electrolyte_source_nodes = np.loadtxt(
            "shape_func_value_electrolyte_source_nodes.txt"
        )
        shape_func_source_nodes_electrolyte = csc_matrix(
            (
                np.array(shape_func_value_electrolyte_source_nodes),
                (
                    np.array(phi_nonzero_index_row_electrolyte_source_nodes),
                    np.array(phi_nonzero_index_column_electrolyte_source_nodes),
                ),
            ),
            shape=(np.shape(point_source)[0], num_nodes_electrolyte),
        )
        print(
            "shape function times 1 on line",
            np.max(np.sum(shape_func_source_nodes_electrolyte, axis=1)),
            np.min(np.sum(shape_func_source_nodes_electrolyte, axis=1)),
        )

        M_electrode_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )
        M_P_x_electrode_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )  # partial M partial x
        M_P_y_electrode_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )  # partial M partial y
        (
            phi_nonzero_index_row_electrode_source_nodes,
            phi_nonzero_index_column_electrode_source_nodes,
            phi_nonzerovalue_data_electrode_source_nodes,
            phi_P_x_nonzerovalue_data_electrode_source_nodes,
            phi_P_y_nonzerovalue_data_electrode_source_nodes,
            phi_P_z_nonzerovalue_data_electrode_source_nodes,
            M_electrode_source_nodes,
            M_P_x_electrode_source_nodes,
            M_P_y_electrode_source_nodes,
            M_P_z_electrode_source_nodes,
        ) = compute_phi_M(
            point_source,
            Gauss_grain_id_electrode,
            x_nodes_electrode,
            nodes_grain_id_electrode,
            a_electrode,
            M_electrode_source_nodes,
            M_P_x_electrode_source_nodes,
            M_P_y_electrode_source_nodes,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        np.savetxt(
            "phi_nonzero_index_row_electrode_source_nodes.txt",
            phi_nonzero_index_row_electrode_source_nodes,
        )
        np.savetxt(
            "phi_nonzero_index_column_electrode_source_nodes.txt",
            phi_nonzero_index_column_electrode_source_nodes,
        )
        phi_nonzero_index_row_electrode_source_nodes = np.loadtxt(
            "phi_nonzero_index_row_electrode_source_nodes.txt"
        )
        phi_nonzero_index_column_electrode_source_nodes = np.loadtxt(
            "phi_nonzero_index_column_electrode_source_nodes.txt"
        )
        num_non_zero_phi_a_electrode_source_nodes = np.shape(
            np.array(phi_nonzero_index_row_electrode_source_nodes)
        )[0]
        shape_func_value_electrode_source_nodes = shape_func_n_nodes_by_n_nodes(
            point_source,
            x_nodes_electrode,
            num_non_zero_phi_a_electrode_source_nodes,
            HT0,
            M_electrode_source_nodes,
            phi_nonzerovalue_data_electrode_source_nodes,
            phi_nonzero_index_row_electrode_source_nodes,
            phi_nonzero_index_column_electrode_source_nodes,
        )
        np.savetxt(
            "shape_func_value_electrode_source_nodes.txt",
            shape_func_value_electrode_source_nodes,
        )
        shape_func_value_electrode_source_nodes = np.loadtxt(
            "shape_func_value_electrode_source_nodes.txt"
        )
        shape_func_source_nodes_electrode = csc_matrix(
            (
                np.array(shape_func_value_electrode_source_nodes),
                (
                    np.array(phi_nonzero_index_row_electrode_source_nodes),
                    np.array(phi_nonzero_index_column_electrode_source_nodes),
                ),
            ),
            shape=(np.shape(point_source)[0], num_nodes_electrode),
        )
        print(
            "shape function times 1 on line",
            np.max(np.sum(shape_func_source_nodes_electrode, axis=1)),
            np.min(np.sum(shape_func_source_nodes_electrode, axis=1)),
        )

        M_pore_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )
        M_P_x_pore_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )  # partial M partial x
        M_P_y_pore_source_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_source)[0])]
        )  # partial M partial y
        (
            phi_nonzero_index_row_pore_source_nodes,
            phi_nonzero_index_column_pore_source_nodes,
            phi_nonzerovalue_data_pore_source_nodes,
            phi_P_x_nonzerovalue_data_pore_source_nodes,
            phi_P_y_nonzerovalue_data_pore_source_nodes,
            phi_P_z_nonzerovalue_data_pore_source_nodes,
            M_pore_source_nodes,
            M_P_x_pore_source_nodes,
            M_P_y_pore_source_nodes,
            M_P_z_pore_source_nodes,
        ) = compute_phi_M(
            point_source,
            Gauss_grain_id_pore,
            x_nodes_pore,
            nodes_grain_id_pore,
            a_pore,
            M_pore_source_nodes,
            M_P_x_pore_source_nodes,
            M_P_y_pore_source_nodes,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        np.savetxt(
            "phi_nonzero_index_row_pore_source_nodes.txt",
            phi_nonzero_index_row_pore_source_nodes,
        )
        np.savetxt(
            "phi_nonzero_index_column_pore_source_nodes.txt",
            phi_nonzero_index_column_pore_source_nodes,
        )
        phi_nonzero_index_row_pore_source_nodes = np.loadtxt(
            "phi_nonzero_index_row_pore_source_nodes.txt"
        )
        phi_nonzero_index_column_pore_source_nodes = np.loadtxt(
            "phi_nonzero_index_column_pore_source_nodes.txt"
        )
        num_non_zero_phi_a_pore_source_nodes = np.shape(
            np.array(phi_nonzero_index_row_pore_source_nodes)
        )[0]
        shape_func_value_pore_source_nodes = shape_func_n_nodes_by_n_nodes(
            point_source,
            x_nodes_pore,
            num_non_zero_phi_a_pore_source_nodes,
            HT0,
            M_pore_source_nodes,
            phi_nonzerovalue_data_pore_source_nodes,
            phi_nonzero_index_row_pore_source_nodes,
            phi_nonzero_index_column_pore_source_nodes,
        )
        np.savetxt(
            "shape_func_value_pore_source_nodes.txt", shape_func_value_pore_source_nodes
        )
        shape_func_value_pore_source_nodes = np.loadtxt(
            "shape_func_value_pore_source_nodes.txt"
        )
        shape_func_source_nodes_pore = csc_matrix(
            (
                np.array(shape_func_value_pore_source_nodes),
                (
                    np.array(phi_nonzero_index_row_pore_source_nodes),
                    np.array(phi_nonzero_index_column_pore_source_nodes),
                ),
            ),
            shape=(np.shape(point_source)[0], num_nodes_pore),
        )
        print(
            "shape function times 1 on line",
            np.max(np.sum(shape_func_source_nodes_pore, axis=1)),
            np.min(np.sum(shape_func_source_nodes_pore, axis=1)),
        )

        M_fixed_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_fixed)[0])]
        )
        M_P_x_fixed_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_fixed)[0])]
        )  # partial M partial x
        M_P_y_fixed_nodes = np.array(
            [np.zeros((3, 3)) for _ in range(np.shape(point_fixed)[0])]
        )  # partial M partial y
        (
            phi_nonzero_index_row_fixed_nodes,
            phi_nonzero_index_column_fixed_nodes,
            phi_nonzerovalue_data_fixed_nodes,
            phi_P_x_nonzerovalue_data_fixed_nodes,
            phi_P_y_nonzerovalue_data_fixed_nodes,
            phi_P_z_nonzerovalue_data_fixed_nodes,
            M_fixed_nodes,
            M_P_x_fixed_nodes,
            M_P_y_fixed_nodes,
            M_P_z_fixed_nodes,
        ) = compute_phi_M(
            point_fixed,
            Gauss_grain_id_mechanical,
            x_nodes_mechanical,
            nodes_grain_id_mechanical,
            a_mechanical,
            M_fixed_nodes,
            M_P_x_fixed_nodes,
            M_P_y_fixed_nodes,
            num_interface_segments,
            interface_nodes,
            BxByCxCy,
            IM_RKPM,
            single_grain,
        )
        np.savetxt(
            "phi_nonzero_index_row_fixed_nodes.txt", phi_nonzero_index_row_fixed_nodes
        )
        np.savetxt(
            "phi_nonzero_index_column_fixed_nodes.txt",
            phi_nonzero_index_column_fixed_nodes,
        )
        phi_nonzero_index_row_fixed_nodes = np.loadtxt(
            "phi_nonzero_index_row_fixed_nodes.txt"
        )
        phi_nonzero_index_column_fixed_nodes = np.loadtxt(
            "phi_nonzero_index_column_fixed_nodes.txt"
        )
        num_non_zero_phi_a_fixed_nodes = np.shape(
            np.array(phi_nonzero_index_row_fixed_nodes)
        )[0]
        det_J_time_weight_fixed_nodes = np.ones(np.shape(point_fixed)[0])
        (
            shape_func_value_fixed_point,
            shape_func_times_det_J_time_weight_value_fixed_point,
            grad_shape_func_x_value_fixed_point,
            grad_shape_func_y_value_fixed_point,
            grad_shape_func_z_value_fixed_point,
            grad_shape_func_x_times_det_J_time_weight_value_fixed_point,
            grad_shape_func_y_times_det_J_time_weight_value_fixed_point,
            grad_shape_func_z_times_det_J_time_weight_value_fixed_point,
        ) = shape_grad_shape_func(
            point_fixed,
            x_nodes_mechanical,
            num_non_zero_phi_a_fixed_nodes,
            HT0,
            M_fixed_nodes,
            M_P_x_fixed_nodes,
            M_P_y_fixed_nodes,
            differential_method,
            HT1,
            HT2,
            phi_nonzerovalue_data_fixed_nodes,
            phi_P_x_nonzerovalue_data_fixed_nodes,
            phi_P_y_nonzerovalue_data_fixed_nodes,
            phi_nonzero_index_row_fixed_nodes,
            phi_nonzero_index_column_fixed_nodes,
            det_J_time_weight_fixed_nodes,
            IM_RKPM,
        )
        np.savetxt("shape_func_value_fixed_point.txt", shape_func_value_fixed_point)
        np.savetxt(
            "grad_shape_func_x_value_fixed_point.txt",
            grad_shape_func_x_value_fixed_point,
        )
        np.savetxt(
            "grad_shape_func_y_value_fixed_point.txt",
            grad_shape_func_y_value_fixed_point,
        )
        shape_func_value_fixed_point = np.loadtxt("shape_func_value_fixed_point.txt")
        grad_shape_func_x_value_fixed_point = np.loadtxt(
            "grad_shape_func_x_value_fixed_point.txt"
        )
        grad_shape_func_y_value_fixed_point = np.loadtxt(
            "grad_shape_func_y_value_fixed_point.txt"
        )
        shape_func_fixed_point = csc_matrix(
            (
                np.array(shape_func_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_nodes),
                    np.array(phi_nonzero_index_column_fixed_nodes),
                ),
            ),
            shape=(np.shape(point_fixed)[0], num_nodes_mechanical),
        )
        grad_shape_func_x_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_x_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_nodes),
                    np.array(phi_nonzero_index_column_fixed_nodes),
                ),
            ),
            shape=(np.shape(point_fixed)[0], num_nodes_mechanical),
        )
        grad_shape_func_y_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_y_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_nodes),
                    np.array(phi_nonzero_index_column_fixed_nodes),
                ),
            ),
            shape=(np.shape(point_fixed)[0], num_nodes_mechanical),
        )

    """
    1. shape function used to interpolate the phi and phie at the interface line (3d), shape: number of gauss points on source line times number of nodes
    2. shape function used to interpolate the displacement at the fixed line (3d), shape: number of gauss points on fixed line times number of nodes
    """
    if dimention == 3:
        M_electrolyte_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )
        M_P_x_electrolyte_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial x
        M_P_y_electrolyte_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial y
        M_P_z_electrolyte_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial y

        # phi_nonzero_index_row_electrolyte_line_nodes, phi_nonzero_index_column_electrolyte_line_nodes, phi_nonzerovalue_data_electrolyte_line_nodes, phi_P_x_nonzerovalue_data_electrolyte_line_nodes, phi_P_y_nonzerovalue_data_electrolyte_line_nodes, phi_P_z_nonzerovalue_data_electrolyte_line_nodes, M_electrolyte_line_nodes, M_P_x_electrolyte_line_nodes, M_P_y_electrolyte_line_nodes, M_P_z_electrolyte_line_nodes = compute_phi_M(x_G_b_line, Gauss_grain_id_electrolyte, x_nodes_electrolyte,nodes_grain_id_electrolyte, a_electrolyte, M_electrolyte_line_nodes, M_P_x_electrolyte_line_nodes, M_P_y_electrolyte_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrolyte_line_nodes)

        # np.savetxt('phi_nonzero_index_row_electrolyte_line_nodes.txt', phi_nonzero_index_row_electrolyte_line_nodes)
        # np.savetxt('phi_nonzero_index_column_electrolyte_line_nodes.txt', phi_nonzero_index_column_electrolyte_line_nodes)
        phi_nonzero_index_row_electrolyte_line_nodes = np.loadtxt(
            "phi_nonzero_index_row_electrolyte_line_nodes.txt"
        )
        phi_nonzero_index_column_electrolyte_line_nodes = np.loadtxt(
            "phi_nonzero_index_column_electrolyte_line_nodes.txt"
        )

        num_non_zero_phi_a_electrolyte_line_nodes = np.shape(
            np.array(phi_nonzero_index_row_electrolyte_line_nodes)
        )[0]

        # shape_func_value_electrolyte_line_nodes = shape_func_n_nodes_by_n_nodes(x_G_b_line,x_nodes_electrolyte, num_non_zero_phi_a_electrolyte_line_nodes,HT0, M_electrolyte_line_nodes, phi_nonzerovalue_data_electrolyte_line_nodes,phi_nonzero_index_row_electrolyte_line_nodes, phi_nonzero_index_column_electrolyte_line_nodes)

        # np.savetxt('shape_func_value_electrolyte_line_nodes.txt', np.asarray(shape_func_value_electrolyte_line_nodes))
        shape_func_value_electrolyte_line_nodes = np.loadtxt(
            "shape_func_value_electrolyte_line_nodes.txt"
        )

        # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
        shape_func_line_n_nodes_electrolyte = csc_matrix(
            (
                np.array(shape_func_value_electrolyte_line_nodes),
                (
                    np.array(phi_nonzero_index_row_electrolyte_line_nodes),
                    np.array(phi_nonzero_index_column_electrolyte_line_nodes),
                ),
            ),
            shape=(num_source_line_gauss_points, num_nodes_electrolyte),
        )

        shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight = (
            shape_func_line_n_nodes_electrolyte.copy()
        )
        shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight.data *= (
            det_J_b_time_weight_line[
                shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight.indices
            ]
        )

        M_electrode_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )
        M_P_x_electrode_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial x
        M_P_y_electrode_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial y
        M_P_z_electrode_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial y

        # phi_nonzero_index_row_electrode_line_nodes, phi_nonzero_index_column_electrode_line_nodes, phi_nonzerovalue_data_electrode_line_nodes, phi_P_x_nonzerovalue_data_electrode_line_nodes, phi_P_y_nonzerovalue_data_electrode_line_nodes, phi_P_z_nonzerovalue_data_electrode_line_nodes, M_electrode_line_nodes, M_P_x_electrode_line_nodes, M_P_y_electrode_line_nodes, M_P_z_electrode_line_nodes = compute_phi_M(x_G_b_line, Gauss_grain_id_electrode, x_nodes_electrode,nodes_grain_id_electrode, a_electrode, M_electrode_line_nodes, M_P_x_electrode_line_nodes, M_P_y_electrode_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_electrode_line_nodes)

        # np.savetxt('phi_nonzero_index_row_electrode_line_nodes.txt', phi_nonzero_index_row_electrode_line_nodes)
        # np.savetxt('phi_nonzero_index_column_electrode_line_nodes.txt', phi_nonzero_index_column_electrode_line_nodes)
        phi_nonzero_index_row_electrode_line_nodes = np.loadtxt(
            "phi_nonzero_index_row_electrode_line_nodes.txt"
        )
        phi_nonzero_index_column_electrode_line_nodes = np.loadtxt(
            "phi_nonzero_index_column_electrode_line_nodes.txt"
        )

        num_non_zero_phi_a_electrode_line_nodes = np.shape(
            np.array(phi_nonzero_index_row_electrode_line_nodes)
        )[0]

        # shape_func_value_electrode_line_nodes = shape_func_n_nodes_by_n_nodes(x_G_b_line,x_nodes_electrode, num_non_zero_phi_a_electrode_line_nodes,HT0, M_electrode_line_nodes, phi_nonzerovalue_data_electrode_line_nodes,phi_nonzero_index_row_electrode_line_nodes, phi_nonzero_index_column_electrode_line_nodes)

        # np.savetxt('shape_func_value_electrode_line_nodes.txt', np.asarray(shape_func_value_electrode_line_nodes))
        shape_func_value_electrode_line_nodes = np.loadtxt(
            "shape_func_value_electrode_line_nodes.txt"
        )

        # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
        shape_func_line_n_nodes_electrode = csc_matrix(
            (
                np.array(shape_func_value_electrode_line_nodes),
                (
                    np.array(phi_nonzero_index_row_electrode_line_nodes),
                    np.array(phi_nonzero_index_column_electrode_line_nodes),
                ),
            ),
            shape=(num_source_line_gauss_points, num_nodes_electrode),
        )
        shape_func_line_n_nodes_electrode_times_det_J_b_time_weight = (
            shape_func_line_n_nodes_electrode.copy()
        )
        shape_func_line_n_nodes_electrode_times_det_J_b_time_weight.data *= (
            det_J_b_time_weight_line[
                shape_func_line_n_nodes_electrode_times_det_J_b_time_weight.indices
            ]
        )

        # print(np.max(np.sum(shape_func_n_nodes_n_nodes_electrolyte.multiply(x_nodes_electrolyte[:,1]),axis=1)-x_nodes_electrolyte), np.min(np.sum(shape_func_n_nodes_n_nodes_electrolyte.multiply(x_nodes_electrolyte[:,1]),axis=1)-x_nodes_electrolyte[:,1]))
        print(
            "shape function times 1 on line",
            np.max(np.sum(shape_func_line_n_nodes_electrode, axis=1)),
            np.min(np.sum(shape_func_line_n_nodes_electrode, axis=1)),
        )

        M_pore_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )
        M_P_x_pore_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial x
        M_P_y_pore_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial y
        M_P_z_pore_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_source_line_gauss_points)]
        )  # partial M partial y

        # phi_nonzero_index_row_pore_line_nodes, phi_nonzero_index_column_pore_line_nodes, phi_nonzerovalue_data_pore_line_nodes, phi_P_x_nonzerovalue_data_pore_line_nodes, phi_P_y_nonzerovalue_data_pore_line_nodes, phi_P_z_nonzerovalue_data_pore_line_nodes, M_pore_line_nodes, M_P_x_pore_line_nodes, M_P_y_pore_line_nodes, M_P_z_pore_line_nodes = compute_phi_M(x_G_b_line, Gauss_grain_id_pore, x_nodes_pore,nodes_grain_id_pore, a_pore, M_pore_line_nodes, M_P_x_pore_line_nodes, M_P_y_pore_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_pore_line_nodes)

        # np.savetxt('phi_nonzero_index_row_pore_line_nodes.txt', phi_nonzero_index_row_pore_line_nodes)
        # np.savetxt('phi_nonzero_index_column_pore_line_nodes.txt', phi_nonzero_index_column_pore_line_nodes)
        phi_nonzero_index_row_pore_line_nodes = np.loadtxt(
            "phi_nonzero_index_row_pore_line_nodes.txt"
        )
        phi_nonzero_index_column_pore_line_nodes = np.loadtxt(
            "phi_nonzero_index_column_pore_line_nodes.txt"
        )

        num_non_zero_phi_a_pore_line_nodes = np.shape(
            np.array(phi_nonzero_index_row_pore_line_nodes)
        )[0]

        # shape_func_value_pore_line_nodes = shape_func_n_nodes_by_n_nodes(x_G_b_line,x_nodes_pore, num_non_zero_phi_a_pore_line_nodes,HT0, M_pore_line_nodes, phi_nonzerovalue_data_pore_line_nodes,phi_nonzero_index_row_pore_line_nodes, phi_nonzero_index_column_pore_line_nodes)

        # np.savetxt('shape_func_value_pore_line_nodes.txt', np.asarray(shape_func_value_pore_line_nodes))
        shape_func_value_pore_line_nodes = np.loadtxt(
            "shape_func_value_pore_line_nodes.txt"
        )

        # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
        shape_func_line_n_nodes_pore = csc_matrix(
            (
                np.array(shape_func_value_pore_line_nodes),
                (
                    np.array(phi_nonzero_index_row_pore_line_nodes),
                    np.array(phi_nonzero_index_column_pore_line_nodes),
                ),
            ),
            shape=(num_source_line_gauss_points, num_nodes_pore),
        )
        shape_func_line_n_nodes_pore_times_det_J_b_time_weight = (
            shape_func_line_n_nodes_pore.copy()
        )
        shape_func_line_n_nodes_pore_times_det_J_b_time_weight.data *= (
            det_J_b_time_weight_line[
                shape_func_line_n_nodes_pore_times_det_J_b_time_weight.indices
            ]
        )

        M_fixed_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_fixed_line_gauss_points)]
        )
        M_P_x_fixed_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_fixed_line_gauss_points)]
        )  # partial M partial x
        M_P_y_fixed_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_fixed_line_gauss_points)]
        )  # partial M partial y
        M_P_z_fixed_line_nodes = np.array(
            [np.zeros((4, 4)) for _ in range(num_fixed_line_gauss_points)]
        )  # partial M partial y

        # phi_nonzero_index_row_fixed_line_nodes, phi_nonzero_index_column_fixed_line_nodes, phi_nonzerovalue_data_fixed_line_nodes, phi_P_x_nonzerovalue_data_fixed_line_nodes, phi_P_y_nonzerovalue_data_fixed_line_nodes, phi_P_z_nonzerovalue_data_fixed_line_nodes, M_fixed_line_nodes, M_P_x_fixed_line_nodes, M_P_y_fixed_line_nodes, M_P_z_fixed_line_nodes = compute_phi_M(x_G_b_line_fixed, Gauss_grain_id_mechanical, x_nodes_mechanical,nodes_grain_id_mechanical, a_mechanical, M_fixed_line_nodes, M_P_x_fixed_line_nodes, M_P_y_fixed_line_nodes, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z_fixed_line_nodes)

        # np.savetxt('phi_nonzero_index_row_fixed_line_nodes.txt', phi_nonzero_index_row_fixed_line_nodes)
        # np.savetxt('phi_nonzero_index_column_fixed_line_nodes.txt', phi_nonzero_index_column_fixed_line_nodes)
        phi_nonzero_index_row_fixed_line_nodes = np.loadtxt(
            "phi_nonzero_index_row_fixed_line_nodes.txt"
        )
        phi_nonzero_index_column_fixed_line_nodes = np.loadtxt(
            "phi_nonzero_index_column_fixed_line_nodes.txt"
        )

        num_non_zero_phi_a_fixed_line_nodes = np.shape(
            np.array(phi_nonzero_index_row_fixed_line_nodes)
        )[0]

        # shape_func_value_fixed_point, shape_func_times_det_J_time_weight_value_fixed_point, grad_shape_func_x_value_fixed_point,grad_shape_func_y_value_fixed_point, grad_shape_func_z_value_fixed_point, grad_shape_func_x_times_det_J_time_weight_value_fixed_point, grad_shape_func_y_times_det_J_time_weight_value_fixed_point, grad_shape_func_z_times_det_J_time_weight_value_fixed_point = shape_grad_shape_func(x_G_b_line_fixed,x_nodes_mechanical, num_non_zero_phi_a_fixed_line_nodes,HT0, M_fixed_line_nodes, M_P_x_fixed_line_nodes, M_P_y_fixed_line_nodes, differential_method, HT1, HT2, phi_nonzerovalue_data_fixed_line_nodes,phi_P_x_nonzerovalue_data_fixed_line_nodes,phi_P_y_nonzerovalue_data_fixed_line_nodes, phi_nonzero_index_row_fixed_line_nodes, phi_nonzero_index_column_fixed_line_nodes, det_J_b_time_weight_line_fixed, IM_RKPM, M_P_z_fixed_line_nodes, HT3, phi_P_z_nonzerovalue_data_fixed_line_nodes)

        # np.savetxt('shape_func_value_fixed_point.txt', np.asarray(shape_func_value_fixed_point))
        # np.savetxt('shape_func_times_det_J_time_weight_value_fixed_point.txt', np.asarray(shape_func_times_det_J_time_weight_value_fixed_point))
        # np.savetxt('grad_shape_func_x_value_fixed_point.txt', np.asarray(grad_shape_func_x_value_fixed_point))
        # np.savetxt('grad_shape_func_y_value_fixed_point.txt', np.asarray(grad_shape_func_y_value_fixed_point))
        # np.savetxt('grad_shape_func_z_value_fixed_point.txt', np.asarray(grad_shape_func_z_value_fixed_point))
        # np.savetxt('grad_shape_func_x_times_det_J_time_weight_value_fixed_point.txt', np.asarray(grad_shape_func_x_times_det_J_time_weight_value_fixed_point))
        # np.savetxt('grad_shape_func_y_times_det_J_time_weight_value_fixed_point.txt', np.asarray(grad_shape_func_y_times_det_J_time_weight_value_fixed_point))
        # np.savetxt('grad_shape_func_z_times_det_J_time_weight_value_fixed_point.txt', np.asarray(grad_shape_func_z_times_det_J_time_weight_value_fixed_point))

        shape_func_value_fixed_point = np.loadtxt("shape_func_value_fixed_point.txt")
        shape_func_times_det_J_time_weight_value_fixed_point = np.loadtxt(
            "shape_func_times_det_J_time_weight_value_fixed_point.txt"
        )
        grad_shape_func_x_value_fixed_point = np.loadtxt(
            "grad_shape_func_x_value_fixed_point.txt"
        )
        grad_shape_func_y_value_fixed_point = np.loadtxt(
            "grad_shape_func_y_value_fixed_point.txt"
        )
        grad_shape_func_z_value_fixed_point = np.loadtxt(
            "grad_shape_func_z_value_fixed_point.txt"
        )
        grad_shape_func_x_times_det_J_time_weight_value_fixed_point = np.loadtxt(
            "grad_shape_func_x_times_det_J_time_weight_value_fixed_point.txt"
        )
        grad_shape_func_y_times_det_J_time_weight_value_fixed_point = np.loadtxt(
            "grad_shape_func_y_times_det_J_time_weight_value_fixed_point.txt"
        )
        grad_shape_func_z_times_det_J_time_weight_value_fixed_point = np.loadtxt(
            "grad_shape_func_z_times_det_J_time_weight_value_fixed_point.txt"
        )

        # numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
        shape_func_fixed_point = csc_matrix(
            (
                np.array(shape_func_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        shape_func_times_det_J_time_weight_fixed_point = csc_matrix(
            (
                np.array(shape_func_times_det_J_time_weight_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        grad_shape_func_x_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_x_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        grad_shape_func_y_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_y_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        grad_shape_func_x_times_det_J_time_weight_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_x_times_det_J_time_weight_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        grad_shape_func_y_times_det_J_time_weight_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_y_times_det_J_time_weight_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        grad_shape_func_z_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_z_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )
        grad_shape_func_z_times_det_J_time_weight_fixed_point = csc_matrix(
            (
                np.array(grad_shape_func_z_times_det_J_time_weight_value_fixed_point),
                (
                    np.array(phi_nonzero_index_row_fixed_line_nodes),
                    np.array(phi_nonzero_index_column_fixed_line_nodes),
                ),
            ),
            shape=(num_fixed_line_gauss_points, num_nodes_mechanical),
        )


comp_shape_func_grad_shape_func_on_boundaries = time.time()


print(
    "time to compute the shape function and grad of shape function on baoundaries = "
    + "%s seconds"
    % (
        comp_shape_func_grad_shape_func_on_boundaries
        - comp_shape_func_grad_shape_func_in_domain
    )
)

if studied_physics == "fuel cell":
    print("aaa")
    # electrolyte:
    print("check shape function in electrolyte")
    print("shape func times 1 in domain")
    print(
        np.max(np.sum(shape_func_electrolyte, axis=1)),
        np.min(np.sum(shape_func_electrolyte, axis=1)),
    )
    print("shape func times 1 on boundary")
    print(
        np.max(np.sum(shape_func_b_electrolyte, axis=1)),
        np.min(np.sum(shape_func_b_electrolyte, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_x_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_x_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_y_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_y_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
        )
    print("partial shape func partial x times x on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                axis=1,
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte.multiply(x_nodes_electrolyte[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte.multiply(x_nodes_electrolyte[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte.multiply(x_nodes_electrolyte[:, 2]),
                    axis=1,
                )
            ),
        )

    # electrode
    print("check shape function in electrode")
    print("shape func times 1 in domain")
    print(
        np.max(np.sum(shape_func_electrode, axis=1)),
        np.min(np.sum(shape_func_electrode, axis=1)),
    )
    print("shape func times 1 on boundary")
    print(
        np.max(np.sum(shape_func_b_electrode, axis=1)),
        np.min(np.sum(shape_func_b_electrode, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_x_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_x_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_y_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_y_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_electrode.multiply(x_nodes_electrode[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_electrode.multiply(x_nodes_electrode[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_electrode.multiply(x_nodes_electrode[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_electrode.multiply(x_nodes_electrode[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
        )

    print("partial shape func partial x times x on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial x times y on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
    )
    print("partial shape func partial y times x on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrode.multiply(x_nodes_electrode[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial y times y on boundary")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrode.multiply(x_nodes_electrode[:, 1]), axis=1
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode.multiply(x_nodes_electrode[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode.multiply(x_nodes_electrode[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode.multiply(x_nodes_electrode[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode.multiply(x_nodes_electrode[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z on boundary")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode.multiply(x_nodes_electrode[:, 2]),
                    axis=1,
                )
            ),
        )

    # pore:
    print("check shape function in pore")
    print("shape func times 1 in domain")
    print(
        np.max(np.sum(shape_func_pore, axis=1)), np.min(np.sum(shape_func_pore, axis=1))
    )
    print("shape func times 1 on boundary")
    print(
        np.max(np.sum(shape_func_b_pore, axis=1)),
        np.min(np.sum(shape_func_b_pore, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(np.sum(grad_shape_func_x_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
        np.min(np.sum(grad_shape_func_x_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(np.sum(grad_shape_func_x_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
        np.min(np.sum(grad_shape_func_x_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(np.sum(grad_shape_func_y_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
        np.min(np.sum(grad_shape_func_y_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(np.sum(grad_shape_func_y_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
        np.min(np.sum(grad_shape_func_y_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(np.sum(grad_shape_func_x_pore.multiply(x_nodes_pore[:, 2]), axis=1)),
            np.min(np.sum(grad_shape_func_x_pore.multiply(x_nodes_pore[:, 2]), axis=1)),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(np.sum(grad_shape_func_y_pore.multiply(x_nodes_pore[:, 2]), axis=1)),
            np.min(np.sum(grad_shape_func_y_pore.multiply(x_nodes_pore[:, 2]), axis=1)),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(np.sum(grad_shape_func_z_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
            np.min(np.sum(grad_shape_func_z_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(np.sum(grad_shape_func_z_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
            np.min(np.sum(grad_shape_func_z_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(np.sum(grad_shape_func_z_pore.multiply(x_nodes_pore[:, 2]), axis=1)),
            np.min(np.sum(grad_shape_func_z_pore.multiply(x_nodes_pore[:, 2]), axis=1)),
        )

    print("partial shape func partial x times x on boundary")
    print(
        np.max(np.sum(grad_shape_func_b_x_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
        np.min(np.sum(grad_shape_func_b_x_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
    )
    print("partial shape func partial x times y on boundary")
    print(
        np.max(np.sum(grad_shape_func_b_x_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
        np.min(np.sum(grad_shape_func_b_x_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
    )
    print("partial shape func partial y times x on boundary")
    print(
        np.max(np.sum(grad_shape_func_b_y_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
        np.min(np.sum(grad_shape_func_b_y_pore.multiply(x_nodes_pore[:, 0]), axis=1)),
    )
    print("partial shape func partial y times y on boundary")
    print(
        np.max(np.sum(grad_shape_func_b_y_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
        np.min(np.sum(grad_shape_func_b_y_pore.multiply(x_nodes_pore[:, 1]), axis=1)),
    )
    if dimention == 3:
        print("partial shape func partial x times z on boundary")
        print(
            np.max(
                np.sum(grad_shape_func_b_x_pore.multiply(x_nodes_pore[:, 2]), axis=1)
            ),
            np.min(
                np.sum(grad_shape_func_b_x_pore.multiply(x_nodes_pore[:, 2]), axis=1)
            ),
        )
        print("partial shape func partial y times z on boundary")
        print(
            np.max(
                np.sum(grad_shape_func_b_y_pore.multiply(x_nodes_pore[:, 2]), axis=1)
            ),
            np.min(
                np.sum(grad_shape_func_b_y_pore.multiply(x_nodes_pore[:, 2]), axis=1)
            ),
        )
        print("partial shape func partial z times x on boundary")
        print(
            np.max(
                np.sum(grad_shape_func_b_z_pore.multiply(x_nodes_pore[:, 0]), axis=1)
            ),
            np.min(
                np.sum(grad_shape_func_b_z_pore.multiply(x_nodes_pore[:, 0]), axis=1)
            ),
        )
        print("partial shape func partial z times y on boundary")
        print(
            np.max(
                np.sum(grad_shape_func_b_z_pore.multiply(x_nodes_pore[:, 1]), axis=1)
            ),
            np.min(
                np.sum(grad_shape_func_b_z_pore.multiply(x_nodes_pore[:, 1]), axis=1)
            ),
        )
        print("partial shape func partial z times z on boundary")
        print(
            np.max(
                np.sum(grad_shape_func_b_z_pore.multiply(x_nodes_pore[:, 2]), axis=1)
            ),
            np.min(
                np.sum(grad_shape_func_b_z_pore.multiply(x_nodes_pore[:, 2]), axis=1)
            ),
        )

    # mechanical:
    print("check shape function mechanical")
    print("shape func times 1 in domain")
    print(
        np.max(np.sum(shape_func_mechanical, axis=1)),
        np.min(np.sum(shape_func_mechanical, axis=1)),
    )

    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_mechanical.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_mechanical.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_mechanical.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_mechanical.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_mechanical.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_mechanical.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_mechanical.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_mechanical.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
    )

    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_x_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_x_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_y_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_y_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_mechanical.multiply(x_nodes_mechanical[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_mechanical.multiply(x_nodes_mechanical[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_mechanical.multiply(x_nodes_mechanical[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_mechanical.multiply(x_nodes_mechanical[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )
    print("check shape function boundary mechanical")
    print("shape func times 1 in domain")
    print(
        np.max(np.sum(shape_func_b_mechanical, axis=1)),
        np.min(np.sum(shape_func_b_mechanical, axis=1)),
    )

    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_mechanical.multiply(x_nodes_mechanical[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_mechanical.multiply(x_nodes_mechanical[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_mechanical.multiply(x_nodes_mechanical[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_mechanical.multiply(x_nodes_mechanical[:, 1]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_mechanical.multiply(x_nodes_mechanical[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_mechanical.multiply(x_nodes_mechanical[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_mechanical.multiply(x_nodes_mechanical[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_mechanical.multiply(x_nodes_mechanical[:, 1]),
                axis=1,
            )
        ),
    )

    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_mechanical.multiply(x_nodes_mechanical[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_mechanical.multiply(x_nodes_mechanical[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_mechanical.multiply(x_nodes_mechanical[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_mechanical.multiply(x_nodes_mechanical[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_mechanical.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

    # fixed nodes
    print("check shape function on fixed node")
    print("shape func times 1 in domain")
    print(
        np.max(np.sum(shape_func_fixed_point, axis=1)),
        np.min(np.sum(shape_func_fixed_point, axis=1)),
    )

    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_fixed_point.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_fixed_point.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_x_fixed_point.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_x_fixed_point.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_fixed_point.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_fixed_point.multiply(x_nodes_mechanical[:, 0]), axis=1
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_y_fixed_point.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_y_fixed_point.multiply(x_nodes_mechanical[:, 1]), axis=1
            )
        ),
    )

    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_x_fixed_point.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_x_fixed_point.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_y_fixed_point.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_y_fixed_point.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_fixed_point.multiply(x_nodes_mechanical[:, 0]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_fixed_point.multiply(x_nodes_mechanical[:, 0]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_fixed_point.multiply(x_nodes_mechanical[:, 1]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_fixed_point.multiply(x_nodes_mechanical[:, 1]),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_z_fixed_point.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_z_fixed_point.multiply(x_nodes_mechanical[:, 2]),
                    axis=1,
                )
            ),
        )

    # source
    if dimention == 2:
        print("check shape function on source node")
        print("shape func times 1 in domain")
        print(
            np.max(np.sum(shape_func_source_nodes_pore, axis=1)),
            np.min(np.sum(shape_func_source_nodes_pore, axis=1)),
        )
        print(
            np.max(np.sum(shape_func_source_nodes_electrolyte, axis=1)),
            np.min(np.sum(shape_func_source_nodes_electrolyte, axis=1)),
        )
        print(
            np.max(np.sum(shape_func_source_nodes_electrode, axis=1)),
            np.min(np.sum(shape_func_source_nodes_electrode, axis=1)),
        )
    if dimention == 3:
        print("check shape function on source node")
        print("shape func times 1 in domain")
        print(
            np.max(np.sum(shape_func_line_n_nodes_electrolyte, axis=1)),
            np.min(np.sum(shape_func_line_n_nodes_electrolyte, axis=1)),
        )
        print(
            np.max(np.sum(shape_func_line_n_nodes_electrode, axis=1)),
            np.min(np.sum(shape_func_line_n_nodes_electrode, axis=1)),
        )
        print(
            np.max(np.sum(shape_func_line_n_nodes_pore, axis=1)),
            np.min(np.sum(shape_func_line_n_nodes_pore, axis=1)),
        )

    # interface
    print("electrolyte-electrode-interface: electolyte")
    print(
        np.max(np.sum(shape_func_b_electrolyte_electrode_electrolyte, axis=1)),
        np.min(np.sum(shape_func_b_electrolyte_electrode_electrolyte, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 0]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 0]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 1]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 1]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 0]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 0]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 1]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrolyte.multiply(
                    x_nodes_electrolyte[:, 1]
                ),
                axis=1,
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrolyte.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

    print("electrolyte-electrode-interface: electrode")
    print(
        np.max(np.sum(shape_func_b_electrolyte_electrode_electrode, axis=1)),
        np.min(np.sum(shape_func_b_electrolyte_electrode_electrode, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrolyte_electrode_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrolyte_electrode_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

    print("electrode-pore-interface: electrode")
    print(
        np.max(np.sum(shape_func_b_electrode_pore_electrode, axis=1)),
        np.min(np.sum(shape_func_b_electrode_pore_electrode, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 0]
                ),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrode_pore_electrode.multiply(
                    x_nodes_electrode[:, 1]
                ),
                axis=1,
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_electrode.multiply(
                        x_nodes_electrode[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

    print("electrode-pore-interface: pore")
    print(
        np.max(np.sum(shape_func_b_electrode_pore_pore, axis=1)),
        np.min(np.sum(shape_func_b_electrode_pore_pore, axis=1)),
    )
    print("partial shape func partial x times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrode_pore_pore.multiply(x_nodes_pore[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrode_pore_pore.multiply(x_nodes_pore[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial x times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_x_electrode_pore_pore.multiply(x_nodes_pore[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_x_electrode_pore_pore.multiply(x_nodes_pore[:, 1]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times x in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrode_pore_pore.multiply(x_nodes_pore[:, 0]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrode_pore_pore.multiply(x_nodes_pore[:, 0]),
                axis=1,
            )
        ),
    )
    print("partial shape func partial y times y in domain")
    print(
        np.max(
            np.sum(
                grad_shape_func_b_y_electrode_pore_pore.multiply(x_nodes_pore[:, 1]),
                axis=1,
            )
        ),
        np.min(
            np.sum(
                grad_shape_func_b_y_electrode_pore_pore.multiply(x_nodes_pore[:, 1]),
                axis=1,
            )
        ),
    )
    if dimention == 3:
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 2]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_electrode_pore_pore.multiply(
                        x_nodes_pore[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

    # n by n
    print("check on domain nodes, n by n")
    print(
        np.max(np.sum(shape_func_n_nodes_n_nodes_electrolyte, axis=1)),
        np.min(np.sum(shape_func_n_nodes_n_nodes_electrolyte, axis=1)),
    )
    print(
        np.max(np.sum(shape_func_n_nodes_n_nodes_electrode, axis=1)),
        np.min(np.sum(shape_func_n_nodes_n_nodes_electrode, axis=1)),
    )
    print(
        np.max(np.sum(shape_func_n_nodes_n_nodes_pore, axis=1)),
        np.min(np.sum(shape_func_n_nodes_n_nodes_pore, axis=1)),
    )

    if dimention == 2 and delta_point_source == "False":
        print("distributed point source")
        print(
            np.max(np.sum(shape_func_b_distributed_point_source_line, axis=1)),
            np.min(np.sum(shape_func_b_distributed_point_source_line, axis=1)),
        )
        print("partial shape func partial x times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial x times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_line.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
        )

    if dimention == 3 and delta_point_source == "False":
        print("distributed point source")
        print(
            np.max(np.sum(shape_func_b_distributed_point_source_surface, axis=1)),
            np.min(np.sum(shape_func_b_distributed_point_source_surface, axis=1)),
        )
        print("partial shape func partial x times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial x times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial x times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_x_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial y times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial y times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_y_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

        print("partial shape func partial z times x in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 0]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times y in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 1]
                    ),
                    axis=1,
                )
            ),
        )
        print("partial shape func partial z times z in domain")
        print(
            np.max(
                np.sum(
                    grad_shape_func_b_z_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
            np.min(
                np.sum(
                    grad_shape_func_b_z_distributed_point_source_surface.multiply(
                        x_nodes_electrolyte[:, 2]
                    ),
                    axis=1,
                )
            ),
        )

print("assemble the matrix and solve all")

###################################################
# assemble the stiffness matrix for mechanical part
###################################################
if studied_physics == "battery":
    # initialize the damage factor D_damage
    D_damage = np.zeros((num_gauss_points_in_domain, 1))

    # initialize the history parameter k
    k = np.ones((num_gauss_points_in_domain, 1)) * k_i

######################
# battery solver
######################

if studied_physics == "battery":

    ini_charge_state = 0.92

    c_ini = (
        np.array(np.ones((num_gauss_points_in_domain, 1))) * c_max * ini_charge_state
    )  # initial concentration at all gauss points, shape:(num_gauss_points_in_domain,1)

    a_lattice_ini, da_lattice_ini = alpha_lattice_complex(
        c_ini / c_max
    )  # initial value of alpha_lattice and dalph_lattice/dx

    c_lattice_ini, dc_lattice_ini = c_lattice_complex(
        c_ini / c_max
    )  # initial value of c_lattice and dc_lattice/dx

    c_n = (
        np.array(np.ones((num_nodes, 1))) * c_max * ini_charge_state
    )  # initial concentration at nodes
    x_n = c_n / c_max  # inital x

    ini_potential = 3.712
    phi_n = np.array(np.ones((num_nodes, 1))) * ini_potential  # initial potential

    dc_threshold = 1.0e-9
    dphi_threshold = 1.0e-9  # when the norm of dc and dphi smaller than the threshold, stop newton iteratio nand move to next time step

    dc = np.array(np.zeros((num_nodes, 1)))
    dphi = np.array(
        np.zeros((num_nodes, 1))
    )  # give an initial value for dc and dphi to start the newton iteration

    c_n1 = c_n + dc  # solutiona from previous newton interation
    phi_n1 = phi_n + dphi

    c_mean_domain = (
        []
    )  # concentration at the point whose index is 'index' at different time
    sigma_x_mean_domain = []
    sigma_y_mean_domain = []
    tau_xy_mean_domain = []
    VM_mean_domain = []
    max_VM_domain = []
    min_VM_domain = []

    time_list = []

    c_min_t = []
    c_max_t = []

    c_min_t_gauss = []
    c_max_t_gauss = []

    D_damage_mean = []
    D_damage_min = []
    D_damage_max = []

    phi_min = []
    phi_max = []
    phi_mean = []

    def_initial = time.time()

    print(
        "time to define initial condition = "
        + "%s seconds" % (def_initial - comp_shape_func_grad_shape_func_on_boundaries)
    )

    K_min = []
    K_max = []
    K_mean = []

    f_min = []
    f_max = []
    f_mean = []

    for ii in range(nt):
        print("time_step:" + str(ii))

        t = dt + dt * ii

        newton_iter_num = 0

        while (
            newton_iter_num < 10
        ):  # (np.linalg.norm(dc)/c_max/ini_charge_state>dc_threshold or np.linalg.norm(dphi)>dphi_threshold) or newton_iter_num==0:

            def_initial = time.time()

            newton_iter_num = newton_iter_num + 1

            Dx, dDx_dx = Dn_complex(shape_func * (c_n1 / c_max), D_damage)

            dDx_dc = dDx_dx / c_max  # diffucivity and dD/dc, size=(n_nodes*n_nodes, 1)

            Dy = Dx / Dx_div_Dy
            dDy_dc = dDx_dc / Dx_div_Dy

            j0, dj0_dx = i_0_complex(shape_func_b * (c_n1 / c_max))
            dj0_dc = dj0_dx / c_max

            E_eq, dE_eq_dx = ocp_complex(shape_func_b * (c_n1 / c_max))
            dE_eq_dc = dE_eq_dx / c_max

            djbv_deta, djbv_dj0, j_BV = i_se(
                shape_func_b * phi_n1, j0, E_eq, Fday, R, Tk
            )  # di/d\eta and dj/dj0 at cn and phi_n;

            #!!!! for all parameters denpend on cencentration or potential, if you want to investigate them at the gauss point, always calculate the concentration
            #!!!! and potential at the gauss point, then use the concentration or potential at gauss point to calculate parameters that depend on concentration and potential.
            #!!!! if you calsulate the parameter at the nodes using concentration and potential at nodes, then interpolate the computed parameters at nodes to gauss point by times the
            #!!!! shape function, it is not accurate!!!!

            # rotate the diffusivity
            R11 = (np.cos(gauss_angle)).reshape(num_gauss_points_in_domain, 1)
            R12 = (np.sin(gauss_angle)).reshape(num_gauss_points_in_domain, 1)
            R21 = (-np.sin(gauss_angle)).reshape(num_gauss_points_in_domain, 1)
            R22 = (np.cos(gauss_angle)).reshape(num_gauss_points_in_domain, 1)

            dD_dc_R11 = (dDx_dc) * (R11**2) + (dDy_dc) * (R12**2)
            dD_dc_R12 = (dDx_dc) * (R11 * R21) + (dDy_dc) * (R12 * R22)
            dD_dc_R21 = (dDx_dc) * (R11 * R21) + (dDy_dc) * (R12 * R22)
            dD_dc_R22 = (dDx_dc) * (R21**2) + (dDy_dc) * (R22**2)

            D_R11 = (Dx) * (R11**2) + (Dy) * (R12**2)
            D_R12 = (Dx) * (R11 * R21) + (Dy) * (R12 * R22)
            D_R21 = (Dx) * (R11 * R21) + (Dy) * (R12 * R22)
            D_R22 = (Dx) * (R21**2) + (Dy) * (
                R22**2
            )  # a sparse matrix dot an array is an array.

            # for a sparse matrix A of shape n*m, if want to times the ith column of A by ith component of 1d array B with shape of m, do A.multiple(B), this returns a sparse matrix
            # if you want to times the ith row of A by ith component of 1d array B with shape of n, you need do scipy.sparse.diags(B).dot(A), which also return a sparse matrix
            # if you want to times the ith row of A by ith component of 1d array B with shape of n*1, you can do A.multiply(B), which also return a sparse matrix

            #########################################
            # define the matrix form for diffusion
            #########################################

            K, f = diffusion_matrix(
                shape_func,
                shape_func_times_det_J_time_weight,
                grad_shape_func_x,
                D_R11,
                grad_shape_func_y,
                D_R21,
                grad_shape_func_x_times_det_J_time_weight,
                D_R12,
                D_R22,
                grad_shape_func_y_times_det_J_time_weight,
                dD_dc_R11,
                dD_dc_R21,
                c_n1,
                dD_dc_R12,
                dD_dc_R22,
                shape_func_b,
                djbv_deta,
                dE_eq_dc,
                shape_func_b_times_det_J_b_time_weight,
                djbv_dj0,
                dj0_dc,
                j_BV,
                j_applied,
                x_G_b,
                k_con,
                dt,
                Fday,
                c_n,
                phi_n1,
            )

            def_diffusion_matrix = time.time()
            print(
                "time to define diffusion matrix= "
                + "%s seconds" % (def_diffusion_matrix - def_initial)
            )

            #########################################
            # calculate the value at RPK nodes
            #########################################

            du = spsolve(K, f)
            # du = np.dot(np.linalg.inv(K.toarray()), f)

            dc[:, 0] = du[0:num_nodes]
            dphi[:, 0] = du[num_nodes:]

            print(
                "Number of Newton Iteration: " + str(newton_iter_num),
                "in Time Step: " + str(ii),
                np.linalg.norm(dc) / (c_max * ini_charge_state),
                np.linalg.norm(dphi),
            )

            c_n1 = c_n1 + dc
            phi_n1 = (
                phi_n1 + dphi
            )  # tentative c and phi for n+1 time step for next newton iteration

            # if ii==0 and newton_iter_num==1:
            solv_diffusion = time.time()
            print(
                "time to solve and update diffusion matrix= "
                + "%s seconds" % (solv_diffusion - def_diffusion_matrix)
            )

        c_n[:, 0] = c_n1[:, 0]
        phi_n[:, 0] = phi_n1[
            :, 0
        ]  # update the concentration and potential after each time step

        ##################################################
        # evaluate the predicted value at all gauss points
        ##################################################
        c_G_domain, c_G_boundary = evaluate_at_gauss_points(
            shape_func.toarray(), shape_func_b.toarray(), c_n
        )
        phi_G_domain, phi_G_boundary = evaluate_at_gauss_points(
            shape_func.toarray(), shape_func_b.toarray(), phi_n
        )

        phi_mean.append(np.mean(phi_G_domain))
        phi_max.append(np.max(phi_G_domain))
        phi_min.append(np.min(phi_G_domain))

        c_mean_domain.append(np.mean(c_G_domain))

        c_min_t.append(np.min(c_n))
        c_max_t.append(np.max(c_n))

        c_min_t_gauss.append(np.min(c_G_domain))
        c_max_t_gauss.append(np.max(c_G_domain))

        time_list.append(t)

        K_min.append(np.min(K))
        K_max.append(np.max(K))
        K_mean.append(np.mean(K))

        f_min.append(np.min(f))
        f_max.append(np.max(f))
        f_mean.append(np.mean(f))

        ####################################################################
        # assemble matrix for mechanical simulation and solve
        ####################################################################
        # if ii==0:
        start_mechanical_time = time.time()

        print("define mechanical stiffness matrix")

        C11, C12, C13, C22, C23, C33 = mechanical_C_tensor(
            num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle
        )

        K_mechanical = mechanical_stiffness_matrix_battery(
            C11,
            C12,
            C13,
            C22,
            C23,
            C33,
            num_gauss_points_in_domain,
            grad_shape_func_x_times_det_J_time_weight,
            grad_shape_func_x,
            grad_shape_func_y_times_det_J_time_weight,
            grad_shape_func_y,
        )

        comp_mechanical_stiffness_matrix = time.time()

        print(
            "time to compute the mechanical stiffness matrix = "
            + "%s seconds" % (comp_mechanical_stiffness_matrix - start_mechanical_time)
        )

        # compute Beta1
        a_lattice, da_lattice_dx = alpha_lattice_complex(c_G_domain / c_max)
        delta_a_lattice = a_lattice - a_lattice_ini
        delta_c = c_G_domain - c_ini
        beta_1 = (
            delta_a_lattice / a_lattice_ini
        )  # all of beta_1 and delta_c are np array
        # compute beta_2
        c_lattice, dc_lattice_dx = c_lattice_complex(c_G_domain / c_max)
        delta_c_lattice = c_lattice - c_lattice_ini
        beta_2 = (
            delta_c_lattice / c_lattice_ini
        )  # all of beta_2 and delta_c are np array

        epsilon_D1 = R11**2 * beta_1 + R12**2 * beta_2
        epsilon_D2 = R21**2 * beta_1 + R22**2 * beta_2
        epsilon_D3 = R21 * R11 * 2 * beta_1 + R22 * R12 * 2 * beta_2

        # solve the mechenical part without damage
        f_mechanical = mechanical_force_matrix(
            x_G,
            C11,
            C12,
            C13,
            C22,
            C23,
            C33,
            epsilon_D1,
            epsilon_D2,
            epsilon_D3,
            grad_shape_func_x_times_det_J_time_weight,
            grad_shape_func_y_times_det_J_time_weight,
        )

        # if ii==0:
        comp_mechanical_force_matrix = time.time()

        print(
            "time to compute mechanical force matrix= "
            + "%s seconds"
            % (comp_mechanical_force_matrix - comp_mechanical_stiffness_matrix)
        )

        # solve displacement field
        u_disp = spsolve(K_mechanical, f_mechanical)  # 1d array

        ux = u_disp[0:num_nodes]  # disp at nodes along x
        uy = u_disp[num_nodes:]  # disp at nodes along y

        # ux = u_disp[list(range(0,2*n_nodes*n_nodes,2))]          # disp at nodes along x
        # uy = u_disp[list(range(1,2*n_nodes*n_nodes,2))]           # disp at nodes along y

        ux_gauss = shape_func * ux  # disp at all gauss points along x
        uy_gauss = shape_func * uy  # disp at all gauss points along y

        """
        (grad_shape_func_x*ux) has shape of (number 0f gauss points,), epsilon_D1 has shape of (number of gauss point, 1), if don't reshape, 
        epsilon_x - epsilon_D1 would have shape of (number of gauss point, number of gauss point).
        !!!!! reshape (grad_shape_func_x*ux)
        """
        epsilon_x = (grad_shape_func_x * ux).reshape(
            num_gauss_points_in_domain, 1
        )  # normal strain along x at all gauss points
        epsilon_y = (grad_shape_func_y * uy).reshape(
            num_gauss_points_in_domain, 1
        )  # normal strain along y at all gauss points
        gamma_xy = ((grad_shape_func_x * uy + grad_shape_func_y * ux) * 0.5).reshape(
            num_gauss_points_in_domain, 1
        )  # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array

        epsilon_x_mechanical = epsilon_x - epsilon_D1
        epsilon_y_mechanical = epsilon_y - epsilon_D2
        gamma_xy_mechanical = gamma_xy - epsilon_D3 / 2

        # calculate the principle strain:
        epsilon_1 = (epsilon_x_mechanical + epsilon_y_mechanical) / 2 + (
            ((epsilon_x_mechanical - epsilon_y_mechanical) / 2) ** 2
            + gamma_xy_mechanical**2
        ) ** 0.5
        epsilon_2 = (epsilon_x_mechanical + epsilon_y_mechanical) / 2 - (
            ((epsilon_x_mechanical - epsilon_y_mechanical) / 2) ** 2
            + gamma_xy_mechanical**2
        ) ** 0.5

        epsilon_1[epsilon_1 < 0] = 0
        epsilon_2[epsilon_2 < 0] = 0

        # print('epsilon_1:', np.mean(epsilon_1), np.max(epsilon_1), np.min(epsilon_1))
        # print('epsilon_2:', np.mean(epsilon_2), np.max(epsilon_2), np.min(epsilon_2))

        epsilon_e_eq = (epsilon_1**2 + epsilon_2**2) ** 0.5

        # print('epsilon_eq:', np.mean(epsilon_e_eq), np.max(epsilon_e_eq), np.min(epsilon_e_eq))

        k = np.fmax(epsilon_e_eq, k)

        D_damage[np.logical_and(k > k_i, k <= k_f)] = (
            (k[np.logical_and(k > k_i, k <= k_f)] - k_i)
            / (k_f - k_i)
            * k_f
            / k[np.logical_and(k > k_i, k <= k_f)]
        )
        D_damage[k > k_f] = 1.0
        D_damage[k <= k_i] = 0.0

        if damage_model == "OFF":
            D_damage[:] = 0.0

        # update the damge factor

        C11, C12, C13, C22, C23, C33 = mechanical_C_tensor(
            num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle
        )

        sigma_x = (
            epsilon_x_mechanical * C11
            + epsilon_y_mechanical * C12
            + gamma_xy_mechanical * 2 * C13
        )  # normal stress along x-direction at all gauess points
        sigma_y = (
            epsilon_x_mechanical * C12
            + epsilon_y_mechanical * C22
            + gamma_xy_mechanical * 2 * C23
        )  # normal stress along y-direction at all gauess points
        tau_xy = (
            epsilon_x_mechanical * C13
            + epsilon_y_mechanical * C23
            + gamma_xy_mechanical * 2 * C33
        )  # shear stress at all gauess points

        Von_mises_stress = (
            sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2
        ) ** 0.5

        sigma_x_mean_domain.append(np.mean(sigma_x))
        sigma_y_mean_domain.append(np.mean(sigma_y))
        tau_xy_mean_domain.append(np.mean(tau_xy))
        VM_mean_domain.append(np.mean(Von_mises_stress))
        max_VM_domain.append(np.max(Von_mises_stress))
        min_VM_domain.append(np.min(Von_mises_stress))
        D_damage_mean.append(np.mean(D_damage))
        D_damage_min.append(np.min(D_damage))
        D_damage_max.append(np.max(D_damage))

        if ii == 0:
            comp_solve_mechanical_stress = time.time()
            print(
                "time to solve mechanical = "
                + "%s seconds"
                % (comp_solve_mechanical_stress - comp_mechanical_force_matrix)
            )

    # # ######################################
    # #     post-process
    # # #######################################
    # # print('averaged damage factor on Gauss points: ', D_damage_mean)
    # # print('averaged VM in domain:', VM_mean_domain)
    # # print('averaged concentration on gauss points:', c_mean_domain)
    # # print('maximum concentration on nodes:', c_max_t)
    # # print('minimum concentration on nodes:', c_min_t)
    # # print('maximum concentration on Gauss points:', c_max_t_gauss)
    # # print('minimum concentration on Gauss points:', c_min_t_gauss)
    # # print('maximum VM on Gauss points:', max_VM_domain)
    # # print('minimum VM on Gauss points:', min_VM_domain)

    np.savetxt("ave_damage_G_DM_damage_long127_direct_distance.txt", D_damage_mean)
    np.savetxt("min_damage_G_DM_damage_long127_direct_distance.txt", D_damage_min)
    np.savetxt("max_damage_G_DM_damage_long127_direct_distance.txt", D_damage_max)
    np.savetxt("ave_VM_DM_damage_long127_direct_distance.txt", VM_mean_domain)
    np.savetxt("max_VM_DM_damage_long127_direct_distance.txt", max_VM_domain)
    np.savetxt("min_VM_DM_damage_long127_direct_distance.txt", min_VM_domain)
    np.savetxt("ave_con_DM_damage_long127_direct_distance.txt", c_mean_domain)
    np.savetxt("max_con_DM_damage_long127_direct_distance.txt", c_max_t_gauss)
    np.savetxt("min_con_DM_damage_long127_direct_distance.txt", c_min_t_gauss)
    np.savetxt("ave_phi_damage_long127_direct_distance.txt", phi_mean)
    np.savetxt("max_phi_damage_long127_direct_distance.txt", phi_max)
    np.savetxt("min_phi_damage_long127_direct_distance.txt", phi_min)

    # _DM_nodamage means the multigrain case with different angles, direct gradient, no damage, only concentration and mechanical simulation,.

    # # fig1 = plt.plot()

    # # plt.plot(time_list, sigma_x_mean_domain, 'b', label = 'Meshfree')

    # # plt.xlabel('time(s)')
    # # plt.ylabel('sigma_x(mean)')
    # # plt.legend()

    # # plt.grid()

    # # # plt.show()
    # # plt.savefig('verify_sigma_x',dpi=300)

    # # fig2 = plt.plot()

    # # plt.plot(time_list, sigma_y_mean_domain, 'b', label = 'Meshfree')

    # # plt.xlabel('time(s)')
    # # plt.ylabel('sigma_y(mean)')
    # # plt.legend()

    # # plt.grid()

    # # # plt.show()
    # # plt.savefig('verify_sigma_y',dpi=300)

    # # fig3 = plt.plot()

    # # plt.plot(time_list, tau_xy_mean_domain, 'b', label = 'Meshfree')

    # # plt.xlabel('time(s)')
    # # plt.ylabel('tau_xy(mean)')
    # # plt.legend()

    # # plt.grid()

    # # # plt.show()
    # # plt.savefig('verify_mean_tau_xy',dpi=300)

    # # fig4 = plt.plot()

    # # plt.plot(time_list, VM_mean_domain, 'b', label = 'Meshfree')

    # # plt.xlabel('time(s)')
    # # plt.ylabel('VM(mean)')
    # # plt.legend()

    # # plt.grid()

    # # # plt.show()
    # # plt.savefig('verify_mean_VM',dpi=300)

    # finish_time = time.time()

    # print('time to assemble matrix and solve equation = ' + "%s seconds" % (finish_time-comp_shape_func_grad_shape_func_on_boundaries))

    # print('total running time = ' + "%s seconds" % (finish_time - start_time))

######################
# fuel cell solver
######################

if studied_physics == "fuel cell":
    if single_grain == "True":
        h_E = (
            y_max_electrolyte - y_min_electrolyte
        ) / n_intervals  # maximum element size
    else:
        h_E = (y_max - y_min) / num_pixels_xyz[1]  # maximum element size

    C_old_electrode = (
        np.array(np.ones((num_nodes_electrode))) * 960
    )  # initial phi is 0.03
    phi_old_electrolyte = 0.8 * np.array(np.ones((num_nodes_electrolyte)))
    C_old_pore = 9.60 * np.array(np.ones((num_nodes_pore)))

    results_old = np.concatenate((phi_old_electrolyte, C_old_electrode, C_old_pore))

    beta_Nitsche_electrode = np.ones((num_nodes_electrode)) * 100 / h_E
    beta_Nitsche_electrolyte = np.ones((num_nodes_electrolyte)) * 100 / h_E
    beta_Nitsche_pore = np.ones((num_nodes_pore)) * 100 / h_E
    beta_Nitsche_mechanical = np.ones((num_nodes_mechanical)) * 100 / h_E

    # when interpolate function g using shape function, g(x)_interpolated = sum over nodes for shape function at x at i_th node times g_i,
    # if g is constant, gi = g = constant, if g is not constant, gi is not g value at the i_th nodes,
    # instead, we need use g(gauss point) = shape_func*gi, gi = shape_func_inverse dot g(gauss point)

    # as we only integral g on left or right boundary where the dirichlet bc are defined, and the value of g on these
    # boundaries are constant, so gi on this boundary nodes can be the correcponding constant, g = shapefunc_b dot gi,
    # g=A is constant on this bondary, so gi=A can satisfy A=shape_func_b dot A.

    # g is zero in electrolyte domain, but not constant in electrode domain
    g_diretchlet_electrolyte = np.zeros(
        (num_gauss_points_on_boundary_electrolyte)
    )  # phi_electolyte(x=0)=g, g=0, at x=0,
    g_diretchlet_electrode = (
        np.ones((num_gauss_points_on_boundary_electrode)) * c_boundary
    )
    g_diretchlet_pore = np.ones((num_gauss_points_on_boundary_pore)) * c_boundary_pore

    # the normal vector (x component) is not constant
    if dimention == 2:
        normal_vector_x_electrolyte = -1
        normal_vector_x_electrode = 1
        normal_vector_x_pore = 1

        normal_vector_y_electrode = 0
        normal_vector_y_electrolyte = 0
        normal_vector_y_pore = 0

        normal_vector_z_electrode = 0
        normal_vector_z_electrolyte = 0
        normal_vector_z_pore = 0
    else:
        normal_vector_x_electrolyte = 0
        normal_vector_x_electrode = 0
        normal_vector_x_pore = 0

        normal_vector_y_electrode = 1
        normal_vector_y_electrolyte = -1
        normal_vector_y_pore = -1

        normal_vector_z_electrode = 0
        normal_vector_z_electrolyte = 0
        normal_vector_z_pore = 0

    if dimention == 2:

        diff = 10.0  # initial difference: 10, if initial_diff<threshold, stop newton interation

        iteration_num = 0

        """
        if the point source is treated as delta function
        
        """
        if delta_point_source == "False":
            phi_old_node_electrolyte = (
                shape_func_source_nodes_electrolyte * phi_old_electrolyte
            )
            i_HOR = i_0 * np.exp(
                0.5 * Fday / R / T * (-phi_old_node_electrolyte + V_app - E_0)
            )

            distributed_point_source_electrolyte = (
                -0.008762
                * np.ones(np.shape(x_G_b_distributed_point_source_line)[0])
                / (y_max - y_min)
                * 2
                * 5
            )  # i_HOR/2###
            point_source_electrode = np.zeros(np.shape(i_HOR)[0])
            point_source_pore = np.zeros(np.shape(i_HOR)[0])
        else:
            # source on triple junctions
            phi_old_node_electrolyte = (
                shape_func_source_nodes_electrolyte * phi_old_electrolyte
            )
            i_HOR = i_0 * np.exp(
                0.5 * Fday / R / T * (-phi_old_node_electrolyte + V_app - E_0)
            )
            # initialize the old results
            point_source_electrolyte_old = i_HOR / 2  ###
            point_source_pore_old = np.zeros(np.shape(i_HOR)[0])  # -i_HOR/96485
            point_source_electrode_old = np.zeros(np.shape(i_HOR)[0])
            # flux will be applied
            point_source_electrolyte = i_HOR / 2  ###
            point_source_pore = np.zeros(np.shape(i_HOR)[0])  # -i_HOR/96485
            point_source_electrode = np.zeros(np.shape(i_HOR)[0])

        # flux across electrolyte/electrode interface
        C_old_electrolyte_electrode = (
            shape_func_b_electrolyte_electrode_electrode * C_old_electrode
        )
        phi_old_electrolyte_electrode = (
            shape_func_b_electrolyte_electrode_electrolyte * phi_old_electrolyte
        )
        i_solid = (
            i_0_solid
            * np.exp(0.5 * Fday / R / T * (-phi_old_electrolyte_electrode + V_app))
            * C_old_electrolyte_electrode
            / c_boundary
        )
        interface_source_electrolyte_electrode_electrolyte_old = -i_solid / 2
        interface_source_electrolyte_electrode_electrode_old = i_solid / 2 / 96485
        interface_source_electrolyte_electrode_electrolyte = -i_solid / 2
        interface_source_electrolyte_electrode_electrode = i_solid / 2 / 96485

        # flux across the electrode/pore interface
        C_old_electrode_pore = shape_func_b_electrode_pore_electrode * C_old_electrode
        J_interface = k_gas * (c_boundary - C_old_electrode_pore)
        interface_source_electrode_pore_electrode_old = -J_interface
        interface_source_electrode_pore_pore_old = J_interface
        interface_source_electrode_pore_electrode = -J_interface
        interface_source_electrode_pore_pore = J_interface

        while diff > 6.0e-7:
            print("iteration number:", iteration_num)

            if delta_point_source == "True":
                K_electrolyte, f_electrolyte = diffusion_matrix_fuel_cell(
                    dimention,
                    point_source_electrolyte,
                    shape_func_source_nodes_electrolyte,
                    g_diretchlet_electrolyte,
                    beta_Nitsche_electrolyte,
                    normal_vector_x_electrolyte,
                    normal_vector_y_electrolyte,
                    diffusion_electrolyte,
                    grad_shape_func_x_electrolyte,
                    grad_shape_func_y_electrolyte,
                    grad_shape_func_x_times_det_J_time_weight_electrolyte,
                    grad_shape_func_y_times_det_J_time_weight_electrolyte,
                    shape_func_b_electrolyte,
                    shape_func_b_times_det_J_b_time_weight_electrolyte,
                    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte,
                    grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte,
                    shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte,
                    interface_source_electrolyte_electrode_electrolyte,
                )
            else:
                K_electrolyte, f_electrolyte = (
                    diffusion_matrix_fuel_cell_distributed_point_source(
                        dimention,
                        distributed_point_source_electrolyte,
                        shape_func_b_times_det_J_b_time_weight_distributed_point_source_line,
                        g_diretchlet_electrolyte,
                        beta_Nitsche_electrolyte,
                        normal_vector_x_electrolyte,
                        normal_vector_y_electrolyte,
                        diffusion_electrolyte,
                        grad_shape_func_x_electrolyte,
                        grad_shape_func_y_electrolyte,
                        grad_shape_func_x_times_det_J_time_weight_electrolyte,
                        grad_shape_func_y_times_det_J_time_weight_electrolyte,
                        shape_func_b_electrolyte,
                        shape_func_b_times_det_J_b_time_weight_electrolyte,
                        grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte,
                        grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte,
                        shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte,
                        interface_source_electrolyte_electrode_electrolyte,
                    )
                )

            interface_source_electrode = np.concatenate(
                (
                    interface_source_electrolyte_electrode_electrode,
                    interface_source_electrode_pore_electrode,
                )
            )
            shape_func_interface_electrode = vstack(
                (
                    shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrode,
                    shape_func_b_times_det_J_b_time_weight_electrode_pore_electrode,
                ),
                format="csc",
            )

            K_electrode, f_electrode = diffusion_matrix_fuel_cell(
                dimention,
                point_source_electrode,
                shape_func_source_nodes_electrode,
                g_diretchlet_electrode,
                beta_Nitsche_electrode,
                normal_vector_x_electrode,
                normal_vector_y_electrode,
                diffusion_electrode,
                grad_shape_func_x_electrode,
                grad_shape_func_y_electrode,
                grad_shape_func_x_times_det_J_time_weight_electrode,
                grad_shape_func_y_times_det_J_time_weight_electrode,
                shape_func_b_electrode,
                shape_func_b_times_det_J_b_time_weight_electrode,
                grad_shape_func_b_x_times_det_J_b_time_weight_electrode,
                grad_shape_func_b_y_times_det_J_b_time_weight_electrode,
                shape_func_interface_electrode,
                interface_source_electrode,
            )

            K_pore, f_pore = diffusion_matrix_fuel_cell(
                dimention,
                point_source_pore,
                shape_func_source_nodes_pore,
                g_diretchlet_pore,
                beta_Nitsche_pore,
                normal_vector_x_pore,
                normal_vector_y_pore,
                diffusion_pore,
                grad_shape_func_x_pore,
                grad_shape_func_y_pore,
                grad_shape_func_x_times_det_J_time_weight_pore,
                grad_shape_func_y_times_det_J_time_weight_pore,
                shape_func_b_pore,
                shape_func_b_times_det_J_b_time_weight_pore,
                grad_shape_func_b_x_times_det_J_b_time_weight_pore,
                grad_shape_func_b_y_times_det_J_b_time_weight_pore,
                shape_func_b_times_det_J_b_time_weight_electrode_pore_pore,
                interface_source_electrode_pore_pore,
            )

            K = block_diag((K_electrolyte, K_electrode, K_pore), format="csc")
            f = np.concatenate((f_electrolyte, f_electrode, f_pore))

            results_new = spsolve(K, f)

            diff = np.linalg.norm(results_new - results_old, 2)
            results_old = results_new.copy()

            iteration_num += 1

            phi_new_electrolyte = results_new[:num_nodes_electrolyte]
            C_new_electrode = results_new[
                num_nodes_electrolyte : (num_nodes_electrolyte + num_nodes_electrode)
            ]
            C_new_pore = results_new[(num_nodes_electrolyte + num_nodes_electrode) :]

            phi_old_electrolyte = results_new[:num_nodes_electrolyte]
            C_old_electrode = results_new[
                num_nodes_electrolyte : (num_nodes_electrolyte + num_nodes_electrode)
            ]
            C_old_pore = results_new[(num_nodes_electrolyte + num_nodes_electrode) :]

            # update the source term
            if delta_point_source == "True":
                # source on triple junctions
                phi_new_node_electrolyte = (
                    shape_func_source_nodes_electrolyte * phi_old_electrolyte
                )
                i_HOR = i_0 * np.exp(
                    0.5 * Fday / R / T * (-phi_new_node_electrolyte + V_app - E_0)
                )
                # initialize the old results
                point_source_electrolyte_new = i_HOR / 2  ###-
                point_source_pore_new = np.zeros(np.shape(i_HOR)[0])  # -i_HOR/96485
                point_source_electrode_new = np.zeros(np.shape(i_HOR)[0])
                # flux will be applied
                point_source_electrolyte = (
                    point_source_electrolyte_new * 0.2
                    + point_source_electrolyte_old * 0.8
                )
                point_source_pore = (
                    point_source_pore_new * 0.2 + point_source_pore_old * 0.8
                )
                point_source_electrode = (
                    point_source_electrode_new * 0.2 + point_source_electrode_old * 0.8
                )
                point_source_electrolyte_old = point_source_electrolyte.copy()
                point_source_pore_old = point_source_pore.copy()
                point_source_electrode_old = point_source_electrode.copy()
            else:
                # distributed point source do not need update the source term
                pass

            # flux across electrolyte/electrode interface
            C_new_electrolyte_electrode = (
                shape_func_b_electrolyte_electrode_electrode * C_old_electrode
            )
            phi_new_electrolyte_electrode = (
                shape_func_b_electrolyte_electrode_electrolyte * phi_old_electrolyte
            )
            i_solid = (
                i_0_solid
                * np.exp(0.5 * Fday / R / T * (-phi_new_electrolyte_electrode + V_app))
                * C_new_electrolyte_electrode
                / c_boundary
            )
            interface_source_electrolyte_electrode_electrolyte_new = -i_solid / 2
            interface_source_electrolyte_electrode_electrode_new = i_solid / 2 / 96485
            interface_source_electrolyte_electrode_electrolyte = (
                interface_source_electrolyte_electrode_electrolyte_new * 0.2
                + interface_source_electrolyte_electrode_electrolyte_old * 0.8
            )
            interface_source_electrolyte_electrode_electrode = (
                interface_source_electrolyte_electrode_electrode_new * 0.2
                + interface_source_electrolyte_electrode_electrode_old * 0.8
            )
            interface_source_electrolyte_electrode_electrolyte_old = (
                interface_source_electrolyte_electrode_electrolyte.copy()
            )
            interface_source_electrolyte_electrode_electrode_old = (
                interface_source_electrolyte_electrode_electrode.copy()
            )

            # flux across the electrode/pore interface
            C_new_electrode_pore = (
                shape_func_b_electrode_pore_electrode * C_old_electrode
            )
            J_interface = k_gas * (c_boundary - C_new_electrode_pore)
            interface_source_electrode_pore_electrode_new = -J_interface
            interface_source_electrode_pore_pore_new = J_interface
            interface_source_electrode_pore_electrode = (
                interface_source_electrode_pore_electrode_new * 0.2
                + interface_source_electrode_pore_electrode_old * 0.8
            )
            interface_source_electrode_pore_pore = (
                interface_source_electrode_pore_pore_new * 0.2
                + interface_source_electrode_pore_pore_old * 0.8
            )
            interface_source_electrode_pore_electrode_old = (
                interface_source_electrode_pore_electrode.copy()
            )
            interface_source_electrode_pore_pore_old = (
                interface_source_electrode_pore_pore.copy()
            )

            print("change of solution", diff)

        potential_on_nodes_save_electrolyte = np.zeros((num_nodes_electrolyte, 3))
        potential_on_nodes_save_electrolyte[:, :2] = x_nodes_electrolyte
        potential_on_nodes_save_electrolyte[:, 2] = phi_new_electrolyte

        potential_on_GP_save_electrolyte = np.zeros(
            (num_gauss_points_in_domain_electrolyte, 3)
        )
        potential_on_GP_save_electrolyte[:, :2] = x_G_electrolyte
        potential_on_GP_save_electrolyte[:, 2] = (
            shape_func_electrolyte * phi_new_electrolyte
        )

        C_on_nodes_save_electrode = np.zeros((num_nodes_electrode, 3))
        C_on_nodes_save_electrode[:, :2] = x_nodes_electrode
        C_on_nodes_save_electrode[:, 2] = C_new_electrode

        C_on_GP_save_electrode = np.zeros((num_gauss_points_in_domain_electrode, 3))
        C_on_GP_save_electrode[:, :2] = x_G_electrode
        C_on_GP_save_electrode[:, 2] = shape_func_electrode * C_new_electrode

        C_on_nodes_save_pore = np.zeros((num_nodes_pore, 3))
        C_on_nodes_save_pore[:, :2] = x_nodes_pore
        C_on_nodes_save_pore[:, 2] = C_new_pore

        C_on_GP_save_pore = np.zeros((num_gauss_points_in_domain_pore, 3))
        C_on_GP_save_pore[:, :2] = x_G_pore
        C_on_GP_save_pore[:, 2] = shape_func_pore * C_new_pore

        fig1 = plt.figure()
        plt.scatter(
            potential_on_GP_save_electrolyte[:, 0],
            potential_on_GP_save_electrolyte[:, 1],
            c=potential_on_GP_save_electrolyte[:, 2],
        )
        plt.colorbar()
        plt.title("potention in electrolyte (on GP)")

        fig2 = plt.figure()
        plt.scatter(
            C_on_GP_save_electrode[:, 0],
            C_on_GP_save_electrode[:, 1],
            c=C_on_GP_save_electrode[:, 2],
        )
        plt.colorbar()
        plt.title("concentration in electrode (on GP)")

        fig3 = plt.figure()
        plt.scatter(
            C_on_GP_save_pore[:, 0], C_on_GP_save_pore[:, 1], c=C_on_GP_save_pore[:, 2]
        )
        plt.colorbar()
        plt.title("concentration in pore (on GP)")

        plt.show()
        exit()

        # mechanical solver:
        ####################################################################
        # assemble matrix for mechanical simulation and solve
        ####################################################################
        # if ii==0:
        start_mechanical_time = time.time()

        D_damage = np.zeros((num_gauss_points_in_domain_mechanical, 1))
        lambda_mechanical_electrode_array = lambda_mechanical_electrode * np.ones(
            num_gauss_points_in_domain_electrode
        )
        lambda_mechanical_electrolyte_array = lambda_mechanical_electrolyte * np.ones(
            num_gauss_points_in_domain_electrolyte
        )
        lambda_mechanical = np.concatenate(
            (lambda_mechanical_electrolyte_array, lambda_mechanical_electrode_array)
        )

        mu_electrode_array = nu_electrode * np.ones(
            num_gauss_points_in_domain_electrode
        )
        mu_electrolyte_array = nu_electrolyte * np.ones(
            num_gauss_points_in_domain_electrolyte
        )
        mu_mechanical = np.concatenate((mu_electrolyte_array, mu_electrode_array))

        print("define mechanical stiffness matrix")

        C11, C12, C13, C22, C23, C33 = mechanical_C_tensor(
            num_gauss_points_in_domain_mechanical,
            D_damage,
            lambda_mechanical,
            mu_mechanical,
            Gauss_angle_mechanical,
        )

        K_mechanical = mechanical_stiffness_matrix_fuel_cell(
            C11,
            C12,
            C13,
            C22,
            C23,
            C33,
            num_gauss_points_in_domain_mechanical,
            grad_shape_func_x_times_det_J_time_weight_mechanical,
            grad_shape_func_x_mechanical,
            grad_shape_func_y_times_det_J_time_weight_mechanical,
            grad_shape_func_y_mechanical,
            beta_Nitsche_mechanical,
            shape_func_b_mechanical,
            shape_func_b_times_det_J_b_time_weight_mechanical,
            grad_shape_func_b_x_times_det_J_b_time_weight_mechanical,
            grad_shape_func_b_y_times_det_J_b_time_weight_mechanical,
            normal_vector_x_electrolyte,
            normal_vector_y_electrolyte,
            shape_func_fixed_point,
            grad_shape_func_x_fixed_point,
            grad_shape_func_y_fixed_point,
        )

        comp_mechanical_stiffness_matrix = time.time()

        print(
            "time to compute the mechanical stiffness matrix = "
            + "%s seconds" % (comp_mechanical_stiffness_matrix - start_mechanical_time)
        )

        dc_G_domain_electrolyte = np.zeros(num_gauss_points_in_domain_electrolyte)
        dc_G_domain = np.concatenate(
            (
                dc_G_domain_electrolyte,
                (shape_func_electrode * C_new_electrode - c_boundary),
            )
        )
        # dc_G_domain = np.ones(num_gauss_points_in_domain_mechanical)*100
        savedata = np.zeros((num_gauss_points_in_domain_mechanical, 3))
        savedata[:, 0] = x_G_mechanical[:, 0]
        savedata[:, 1] = x_G_mechanical[:, 1]
        savedata[:, 2] = dc_G_domain

        # c_G_domain_electrode = np.ones(num_gauss_points_in_domain_electrode)*1020
        # c_G_domain = np.concatenate((c_G_domain_electrolyte, c_G_domain_electrode))
        # fig66 = plt.figure()
        # plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=c_G_domain)
        # plt.colorbar()
        # plt.title('domain C')
        # fig66 = plt.figure()
        # plt.scatter(x_nodes_electrode[:, 0], x_nodes_electrode[:, 1], c=C_new_electrode)
        # plt.colorbar()
        # plt.title('electrode C')
        # plt.show()
        # exit()
        # rotate the diffusivity
        R11 = (np.cos(Gauss_angle_mechanical)).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )
        R12 = (np.sin(Gauss_angle_mechanical)).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )
        R21 = (-np.sin(Gauss_angle_mechanical)).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )
        R22 = (np.cos(Gauss_angle_mechanical)).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )

        """
        # !!!!!!!!!!!!!!!! to debug, set the expantion coefficient be [1,0,0], apply uniform concentration in both electrolyte and electrode
        """

        # compute Beta (expansion coeffieicent)
        beta_fuelcell_expansion_coefficient_electrolyte = (
            beta_fuelcell_expansion_coefficient
            * np.ones(num_gauss_points_in_domain_electrolyte)
        )
        beta_fuelcell_expansion_coefficient_electrode = (
            beta_fuelcell_expansion_coefficient
            * np.ones(num_gauss_points_in_domain_electrode)
        )
        beta_1 = np.concatenate(
            (
                beta_fuelcell_expansion_coefficient_electrolyte,
                beta_fuelcell_expansion_coefficient_electrode,
            )
        ).reshape(num_gauss_points_in_domain_mechanical, 1) * (
            dc_G_domain.reshape(num_gauss_points_in_domain_mechanical, 1)
        )
        beta_2 = np.concatenate(
            (
                beta_fuelcell_expansion_coefficient_electrolyte,
                beta_fuelcell_expansion_coefficient_electrode,
            )
        ).reshape(num_gauss_points_in_domain_mechanical, 1) * (
            dc_G_domain.reshape(num_gauss_points_in_domain_mechanical, 1)
        )

        epsilon_D1 = R11**2 * beta_1 + R12**2 * beta_2
        epsilon_D2 = R21**2 * beta_1 + R22**2 * beta_2
        epsilon_D3 = R21 * R11 * 2 * beta_1 + R22 * R12 * 2 * beta_2

        # fig66 = plt.figure()
        # plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=epsilon_D1)
        # plt.colorbar()
        # plt.title('epsilon_D1')
        # fig66 = plt.figure()
        # plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=epsilon_D2)
        # plt.colorbar()
        # plt.title('epsilon_D2')
        # fig66 = plt.figure()
        # plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=epsilon_D3)
        # plt.colorbar()
        # plt.title('epsilon_D3')
        # plt.show()
        # exit()

        # solve the mechenical part without damage
        f_mechanical = mechanical_force_matrix(
            x_G_mechanical,
            C11,
            C12,
            C13,
            C22,
            C23,
            C33,
            epsilon_D1,
            epsilon_D2,
            epsilon_D3,
            grad_shape_func_x_times_det_J_time_weight_mechanical,
            grad_shape_func_y_times_det_J_time_weight_mechanical,
        )

        # if ii==0:
        comp_mechanical_force_matrix = time.time()

        print(
            "time to compute mechanical force matrix= "
            + "%s seconds"
            % (comp_mechanical_force_matrix - comp_mechanical_stiffness_matrix)
        )

        # solve displacement field
        u_disp = spsolve(K_mechanical, f_mechanical)  # 1d array

        print(np.shape(u_disp), num_nodes_mechanical)

        ux = u_disp[0:num_nodes_mechanical]  # disp at nodes along x
        uy = u_disp[num_nodes_mechanical:]  # disp at nodes along y

        ux_gauss = shape_func_mechanical * ux
        uy_gauss = shape_func_mechanical * uy

        print(
            np.max(x_G_mechanical[:num_gauss_points_in_domain_electrolyte, 0]),
            np.min(x_G_mechanical[:num_gauss_points_in_domain_electrolyte, 0]),
        )
        print(
            np.max(x_G_mechanical[:num_gauss_points_in_domain_electrolyte, 1]),
            np.min(x_G_mechanical[:num_gauss_points_in_domain_electrolyte, 1]),
        )
        print(
            np.max(x_G_mechanical[num_gauss_points_in_domain_electrolyte:, 0]),
            np.min(x_G_mechanical[num_gauss_points_in_domain_electrolyte:, 0]),
        )
        print(
            np.max(x_G_mechanical[num_gauss_points_in_domain_electrolyte:, 1]),
            np.min(x_G_mechanical[num_gauss_points_in_domain_electrolyte:, 1]),
        )
        print(np.max(x_nodes_mechanical[:, 0]), np.min(x_nodes_mechanical[:, 0]))
        print(np.max(x_nodes_mechanical[:, 1]), np.min(x_nodes_mechanical[:, 1]))

        # predict the damage factor:
        """
        (grad_shape_func_x*ux) has shape of (number 0f gauss points,), epsilon_D1 has shape of (number of gauss point, 1), if don't reshape, 
        epsilon_x - epsilon_D1 would have shape of (number of gauss point, number of gauss point).
        !!!!! reshape (grad_shape_func_x*ux)
        """

        epsilon_x = (grad_shape_func_x_mechanical * ux).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # normal strain along x at all gauss points
        epsilon_y = (grad_shape_func_y_mechanical * uy).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # normal strain along y at all gauss points
        gamma_xy = (
            (grad_shape_func_x_mechanical * uy + grad_shape_func_y_mechanical * ux)
            * 0.5
        ).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array

        epsilon_x_mechanical = epsilon_x - epsilon_D1
        epsilon_y_mechanical = epsilon_y - epsilon_D2
        gamma_xy_mechanical = gamma_xy - epsilon_D3 / 2

        # calculate the principle strain:
        epsilon_1 = (epsilon_x_mechanical + epsilon_y_mechanical) / 2 + (
            ((epsilon_x_mechanical - epsilon_y_mechanical) / 2) ** 2
            + gamma_xy_mechanical**2
        ) ** 0.5
        epsilon_2 = (epsilon_x_mechanical + epsilon_y_mechanical) / 2 - (
            ((epsilon_x_mechanical - epsilon_y_mechanical) / 2) ** 2
            + gamma_xy_mechanical**2
        ) ** 0.5

        epsilon_1[epsilon_1 < 0] = 0
        epsilon_2[epsilon_2 < 0] = 0

        # print('epsilon_1:', np.mean(epsilon_1), np.max(epsilon_1), np.min(epsilon_1))
        # print('epsilon_2:', np.mean(epsilon_2), np.max(epsilon_2), np.min(epsilon_2))

        epsilon_e_eq = (epsilon_1**2 + epsilon_2**2) ** 0.5

    if dimention == 3:

        diff = 3000.0  # initial difference: 10, if initial_diff<threshold, stop newton interation

        iteration_num = 0

        """
        if the point source is treated as delta function
        
        """
        if delta_point_source == "False":

            phi_old_line_gauss_electrolyte = (
                shape_func_line_n_nodes_electrolyte * phi_old_electrolyte
            )

            i_HOR = i_0 * np.exp(
                0.5 * Fday / R / T * (-phi_old_line_gauss_electrolyte + V_app - E_0)
            )

            distributed_line_source_electrolyte = (
                -0.0125
                * np.ones(np.shape(x_G_b_distributed_point_source_surface)[0])
                / (z_max - z_min)
                * 2
                * 5
            )  # i_HOR/2###
            line_source_electrode = np.zeros(np.shape(i_HOR)[0])
            line_source_pore = np.zeros(np.shape(i_HOR)[0])
        else:
            phi_old_line_gauss_electrolyte = (
                shape_func_line_n_nodes_electrolyte * phi_old_electrolyte
            )

            i_HOR = i_0 * np.exp(
                0.5 * Fday / R / T * (-phi_old_line_gauss_electrolyte + V_app - E_0)
            )
            line_source_electrolyte_old = (
                i_HOR / 2
            )  # -0.0007852916046055233/2*np.ones(2)# # actually this is on gauss points of interface line, as we need integral across the interface line
            line_source_pore_old = np.zeros(
                np.shape(i_HOR)[0]
            )  # -i_HOR/96485#0.0007852916046055233*np.ones(2)#i_HOR
            line_source_electrode_old = np.zeros(np.shape(i_HOR)[0])
            line_source_electrolyte = (
                i_HOR / 2
            )  # -0.0007852916046055233/2*np.ones(2)# # actually this is on gauss points of interface line, as we need integral across the interface line
            line_source_pore = np.zeros(
                np.shape(i_HOR)[0]
            )  # -i_HOR/96485#0.0007852916046055233*np.ones(2)#i_HOR
            line_source_electrode = np.zeros(np.shape(i_HOR)[0])

        C_old_electrolyte_electrode = (
            shape_func_b_electrolyte_electrode_electrode * C_old_electrode
        )
        phi_old_electrolyte_electrode = (
            shape_func_b_electrolyte_electrode_electrolyte * phi_old_electrolyte
        )
        i_solid = (
            i_0_solid
            * np.exp(0.5 * Fday / R / T * (-phi_old_electrolyte_electrode + V_app))
            * C_old_electrolyte_electrode
            / c_boundary
        )
        interface_source_electrolyte_electrode_electrolyte_old = -i_solid / 2
        interface_source_electrolyte_electrode_electrode_old = i_solid / 2 / 96485
        interface_source_electrolyte_electrode_electrolyte = -i_solid / 2
        interface_source_electrolyte_electrode_electrode = i_solid / 2 / 96485

        C_old_electrode_pore = shape_func_b_electrode_pore_electrode * C_old_electrode
        J_interface = k_gas * (c_boundary - C_old_electrode_pore)
        interface_source_electrode_pore_electrode_old = -J_interface
        interface_source_electrode_pore_pore_old = J_interface
        interface_source_electrode_pore_electrode = -J_interface
        interface_source_electrode_pore_pore = J_interface

        while diff > 6.0e-4:
            print("iteration number:", iteration_num)

            if delta_point_source == "True":

                K_electrolyte, f_electrolyte = diffusion_matrix_fuel_cell(
                    dimention,
                    line_source_electrolyte,
                    shape_func_line_n_nodes_electrolyte_times_det_J_b_time_weight,
                    g_diretchlet_electrolyte,
                    beta_Nitsche_electrolyte,
                    normal_vector_x_electrolyte,
                    normal_vector_y_electrolyte,
                    diffusion_electrolyte,
                    grad_shape_func_x_electrolyte,
                    grad_shape_func_y_electrolyte,
                    grad_shape_func_x_times_det_J_time_weight_electrolyte,
                    grad_shape_func_y_times_det_J_time_weight_electrolyte,
                    shape_func_b_electrolyte,
                    shape_func_b_times_det_J_b_time_weight_electrolyte,
                    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte,
                    grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte,
                    shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte,
                    interface_source_electrolyte_electrode_electrolyte,
                    grad_shape_func_z_electrolyte,
                    grad_shape_func_z_times_det_J_time_weight_electrolyte,
                    grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte,
                    normal_vector_z_electrolyte,
                )
            else:
                K_electrolyte, f_electrolyte = (
                    diffusion_matrix_fuel_cell_distributed_point_source(
                        dimention,
                        distributed_line_source_electrolyte,
                        shape_func_b_times_det_J_b_time_weight_distributed_point_source_surface,
                        g_diretchlet_electrolyte,
                        beta_Nitsche_electrolyte,
                        normal_vector_x_electrolyte,
                        normal_vector_y_electrolyte,
                        diffusion_electrolyte,
                        grad_shape_func_x_electrolyte,
                        grad_shape_func_y_electrolyte,
                        grad_shape_func_x_times_det_J_time_weight_electrolyte,
                        grad_shape_func_y_times_det_J_time_weight_electrolyte,
                        shape_func_b_electrolyte,
                        shape_func_b_times_det_J_b_time_weight_electrolyte,
                        grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte,
                        grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte,
                        shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrolyte,
                        interface_source_electrolyte_electrode_electrolyte,
                        grad_shape_func_z_electrolyte,
                        grad_shape_func_z_times_det_J_time_weight_electrolyte,
                        grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte,
                        normal_vector_z_electrolyte,
                    )
                )

            interface_source_electrode = np.concatenate(
                (
                    interface_source_electrolyte_electrode_electrode,
                    interface_source_electrode_pore_electrode,
                )
            )
            shape_func_interface_electrode = vstack(
                (
                    shape_func_b_times_det_J_b_time_weight_electrolyte_electrode_electrode,
                    shape_func_b_times_det_J_b_time_weight_electrode_pore_electrode,
                ),
                format="csc",
            )

            K_electrode, f_electrode = diffusion_matrix_fuel_cell(
                dimention,
                line_source_electrode,
                shape_func_line_n_nodes_electrode_times_det_J_b_time_weight,
                g_diretchlet_electrode,
                beta_Nitsche_electrode,
                normal_vector_x_electrode,
                normal_vector_y_electrode,
                diffusion_electrode,
                grad_shape_func_x_electrode,
                grad_shape_func_y_electrode,
                grad_shape_func_x_times_det_J_time_weight_electrode,
                grad_shape_func_y_times_det_J_time_weight_electrode,
                shape_func_b_electrode,
                shape_func_b_times_det_J_b_time_weight_electrode,
                grad_shape_func_b_x_times_det_J_b_time_weight_electrode,
                grad_shape_func_b_y_times_det_J_b_time_weight_electrode,
                shape_func_interface_electrode,
                interface_source_electrode,
                grad_shape_func_z_electrode,
                grad_shape_func_z_times_det_J_time_weight_electrode,
                grad_shape_func_b_z_times_det_J_b_time_weight_electrode,
                normal_vector_z_electrode,
            )

            K_pore, f_pore = diffusion_matrix_fuel_cell(
                dimention,
                line_source_pore,
                shape_func_line_n_nodes_pore_times_det_J_b_time_weight,
                g_diretchlet_pore,
                beta_Nitsche_pore,
                normal_vector_x_pore,
                normal_vector_y_pore,
                diffusion_pore,
                grad_shape_func_x_pore,
                grad_shape_func_y_pore,
                grad_shape_func_x_times_det_J_time_weight_pore,
                grad_shape_func_y_times_det_J_time_weight_pore,
                shape_func_b_pore,
                shape_func_b_times_det_J_b_time_weight_pore,
                grad_shape_func_b_x_times_det_J_b_time_weight_pore,
                grad_shape_func_b_y_times_det_J_b_time_weight_pore,
                shape_func_b_times_det_J_b_time_weight_electrode_pore_pore,
                interface_source_electrode_pore_pore,
                grad_shape_func_z_pore,
                grad_shape_func_z_times_det_J_time_weight_pore,
                grad_shape_func_b_z_times_det_J_b_time_weight_pore,
                normal_vector_z_pore,
            )

            K = block_diag((K_electrolyte, K_electrode, K_pore), format="csc")
            f = np.concatenate((f_electrolyte, f_electrode, f_pore))

            results_new = spsolve(K, f)

            diff = np.linalg.norm(results_new - results_old, 2)
            results_old[:] = results_new.copy()

            iteration_num += 1

            phi_new_electrolyte = results_new[:num_nodes_electrolyte]
            C_new_electrode = results_new[
                num_nodes_electrolyte : num_nodes_electrode + num_nodes_electrolyte
            ]
            C_new_pore = results_new[
                num_nodes_electrode
                + num_nodes_electrolyte : num_nodes_electrode
                + num_nodes_electrolyte
                + num_nodes_pore
            ]

            phi_old_electrolyte = results_new[:num_nodes_electrolyte]
            C_old_electrode = results_new[
                num_nodes_electrolyte : num_nodes_electrolyte + num_nodes_electrode
            ]
            C_old_pore = results_new[
                num_nodes_electrolyte
                + num_nodes_electrode : num_nodes_electrolyte
                + num_nodes_electrode
                + num_nodes_pore
            ]
            print("change of solution", diff)

            # update the source term
            if delta_point_source == "True":
                phi_new_line_gauss_electrolyte = (
                    shape_func_line_n_nodes_electrolyte * phi_old_electrolyte
                )
                i_HOR = i_0 * np.exp(
                    0.5 * Fday / R / T * (-phi_new_line_gauss_electrolyte + V_app - E_0)
                )
                line_source_electrolyte_new = (
                    i_HOR / 2
                )  # -0.0007852916046055233/2*np.ones(2)# # actually this is on gauss points of interface line, as we need integral across the interface line
                line_source_pore_new = np.zeros(
                    np.shape(i_HOR)[0]
                )  # -i_HOR/96485#0.0007852916046055233*np.ones(2)#i_HOR
                line_source_electrode_new = np.zeros(np.shape(i_HOR)[0])

                # flux will be applied
                line_source_electrolyte = (
                    line_source_electrolyte_new * 0.5
                    + line_source_electrolyte_old * 0.5
                )
                line_source_pore = (
                    line_source_pore_new * 0.5 + line_source_pore_old * 0.5
                )
                line_source_electrode = (
                    line_source_electrode_new * 0.5 + line_source_electrode_old * 0.5
                )
                line_source_electrolyte_old = line_source_electrolyte.copy()
                line_source_pore_old = line_source_pore.copy()
                line_source_electrode_old = line_source_electrode.copy()
            else:
                # distributed point source do not need update the source term
                pass

            # flux across electrolyte/electrode interface
            C_new_electrolyte_electrode = (
                shape_func_b_electrolyte_electrode_electrode * C_old_electrode
            )
            phi_new_electrolyte_electrode = (
                shape_func_b_electrolyte_electrode_electrolyte * phi_old_electrolyte
            )
            i_solid = (
                i_0_solid
                * np.exp(0.5 * Fday / R / T * (-phi_new_electrolyte_electrode + V_app))
                * C_new_electrolyte_electrode
                / c_boundary
            )
            interface_source_electrolyte_electrode_electrolyte_new = -i_solid / 2
            interface_source_electrolyte_electrode_electrode_new = i_solid / 2 / 96485
            interface_source_electrolyte_electrode_electrolyte = (
                interface_source_electrolyte_electrode_electrolyte_new * 0.5
                + interface_source_electrolyte_electrode_electrolyte_old * 0.5
            )
            interface_source_electrolyte_electrode_electrode = (
                interface_source_electrolyte_electrode_electrode_new * 0.5
                + interface_source_electrolyte_electrode_electrode_old * 0.5
            )
            interface_source_electrolyte_electrode_electrolyte_old = (
                interface_source_electrolyte_electrode_electrolyte.copy()
            )
            interface_source_electrolyte_electrode_electrode_old = (
                interface_source_electrolyte_electrode_electrode.copy()
            )

            # flux across the electrode/pore interface
            C_new_electrode_pore = (
                shape_func_b_electrode_pore_electrode * C_old_electrode
            )
            J_interface = k_gas * (c_boundary - C_new_electrode_pore)
            interface_source_electrode_pore_electrode_new = -J_interface
            interface_source_electrode_pore_pore_new = J_interface
            interface_source_electrode_pore_electrode = (
                interface_source_electrode_pore_electrode_new * 0.5
                + interface_source_electrode_pore_electrode_old * 0.5
            )
            interface_source_electrode_pore_pore = (
                interface_source_electrode_pore_pore_new * 0.5
                + interface_source_electrode_pore_pore_old * 0.5
            )
            interface_source_electrode_pore_electrode_old = (
                interface_source_electrode_pore_electrode.copy()
            )
            interface_source_electrode_pore_pore_old = (
                interface_source_electrode_pore_pore.copy()
            )

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111, projection='3d')
        # sc1 = ax1.scatter(x_G_b_line[:, 0], x_G_b_line[:, 1],x_G_b_line[:, 2], c=line_source_electrolyte)
        # plt.colorbar(sc1, ax=ax1)

        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111, projection='3d')
        # sc2 = ax2.scatter(x_G_b_line[:, 0], x_G_b_line[:, 1],x_G_b_line[:, 2], c=line_source_electrode)
        # plt.colorbar(sc2, ax=ax2)

        # fig3 = plt.figure()
        # ax3 = fig3.add_subplot(111, projection='3d')
        # sc3 = ax3.scatter(x_G_b_line[:, 0], x_G_b_line[:, 1],x_G_b_line[:, 2], c=line_source_pore)
        # plt.colorbar(sc3, ax=ax3)

        # fig4 = plt.figure()
        # ax4 = fig4.add_subplot(111, projection='3d')
        # sc4 = ax4.scatter(x_G_b_interface_electrode_electrolyte[:, 0], x_G_b_interface_electrode_electrolyte[:, 1],x_G_b_interface_electrode_electrolyte[:, 2], c=interface_source_electrolyte_electrode_electrolyte)
        # plt.colorbar(sc4, ax=ax4)
        # fig5 = plt.figure()
        # ax5 = fig5.add_subplot(111, projection='3d')
        # sc5 = ax5.scatter(x_G_b_interface_electrode_electrolyte[:, 0], x_G_b_interface_electrode_electrolyte[:, 1],x_G_b_interface_electrode_electrolyte[:, 2], c=interface_source_electrolyte_electrode_electrode)
        # plt.colorbar(sc5, ax=ax5)

        # fig6 = plt.figure()
        # ax6 = fig6.add_subplot(111, projection='3d')
        # sc6 = ax6.scatter(x_G_b_interface_electrode_pore[:, 0], x_G_b_interface_electrode_pore[:, 1],x_G_b_interface_electrode_pore[:, 2], c=interface_source_electrode_pore_electrode)
        # plt.colorbar(sc6, ax=ax6)

        # fig7 = plt.figure()
        # ax7 = fig7.add_subplot(111, projection='3d')
        # sc7 = ax7.scatter(x_G_b_interface_electrode_pore[:, 0], x_G_b_interface_electrode_pore[:, 1],x_G_b_interface_electrode_pore[:, 2], c=interface_source_electrode_pore_pore)
        # plt.colorbar(sc7, ax=ax7)

        fig8 = plt.figure()
        ax8 = fig8.add_subplot(111, projection="3d")
        sc8 = ax8.scatter(
            x_nodes_electrolyte[:, 0],
            x_nodes_electrolyte[:, 1],
            x_nodes_electrolyte[:, 2],
            c=shape_func_n_nodes_n_nodes_electrolyte * phi_new_electrolyte,
        )
        plt.colorbar(sc8, ax=ax8)

        fig9 = plt.figure()
        ax9 = fig9.add_subplot(111, projection="3d")
        sc9 = ax9.scatter(
            x_nodes_electrode[:, 0],
            x_nodes_electrode[:, 1],
            x_nodes_electrode[:, 2],
            c=shape_func_n_nodes_n_nodes_electrode * C_new_electrode,
        )
        plt.colorbar(sc9, ax=ax9)

        fig10 = plt.figure()
        ax10 = fig10.add_subplot(111, projection="3d")
        sc10 = ax10.scatter(
            x_nodes_pore[:, 0],
            x_nodes_pore[:, 1],
            x_nodes_pore[:, 2],
            c=shape_func_n_nodes_n_nodes_pore * C_new_pore,
        )
        plt.colorbar(sc10, ax=ax10)

        plt.show()

        exit()

        #         iteration number: 0
        # change of solution 77.53718458836529
        # iteration number: 1
        # change of solution 376.5061202281469
        # iteration number: 2
        # change of solution 158.7178289538278
        # iteration number: 3
        # change of solution 107.21865698354357
        # iteration number: 4
        # change of solution 66.91214066976451
        # iteration number: 5
        # change of solution 43.64027670882215
        # iteration number: 6
        # change of solution 27.880517887802174
        # iteration number: 7
        # change of solution 18.148005458395428
        # iteration number: 8
        # change of solution 11.702685655637751
        # iteration number: 9
        # change of solution 7.600799003601473
        # iteration number: 10

        # mechanical solver:
        ####################################################################
        # assemble matrix for mechanical simulation and solve
        ####################################################################
        # if ii==0:
        start_mechanical_time = time.time()

        D_damage = np.zeros((num_gauss_points_in_domain_mechanical, 1))
        lambda_mechanical_electrode_array = lambda_mechanical_electrode * np.ones(
            num_gauss_points_in_domain_electrode
        )
        lambda_mechanical_electrolyte_array = lambda_mechanical_electrolyte * np.ones(
            num_gauss_points_in_domain_electrolyte
        )
        lambda_mechanical = np.concatenate(
            (lambda_mechanical_electrolyte_array, lambda_mechanical_electrode_array)
        )

        mu_electrode_array = nu_electrode * np.ones(
            num_gauss_points_in_domain_electrode
        )
        mu_electrolyte_array = nu_electrolyte * np.ones(
            num_gauss_points_in_domain_electrolyte
        )
        mu_mechanical = np.concatenate((mu_electrolyte_array, mu_electrode_array))

        print("define mechanical stiffness matrix")

        C, T_c = mechanical_C_tensor_3d(
            num_gauss_points_in_domain_mechanical,
            D_damage,
            lambda_mechanical,
            mu_mechanical,
            Gauss_angle_mechanical,
            gauss_rotation_axis,
        )

        K_mechanical = mechanical_stiffness_matrix_3d_fuel_cell(
            C,
            num_gauss_points_in_domain_mechanical,
            grad_shape_func_x_times_det_J_time_weight_mechanical,
            grad_shape_func_x_mechanical,
            grad_shape_func_y_times_det_J_time_weight_mechanical,
            grad_shape_func_y_mechanical,
            grad_shape_func_z_times_det_J_time_weight_mechanical,
            grad_shape_func_z_mechanical,
            beta_Nitsche_mechanical,
            shape_func_fixed_point,
            shape_func_times_det_J_time_weight_fixed_point,
            grad_shape_func_x_fixed_point,
            grad_shape_func_x_times_det_J_time_weight_fixed_point,
            grad_shape_func_y_fixed_point,
            grad_shape_func_y_times_det_J_time_weight_fixed_point,
            grad_shape_func_z_fixed_point,
            grad_shape_func_z_times_det_J_time_weight_fixed_point,
            normal_vector_x_electrolyte,
            normal_vector_y_electrolyte,
            normal_vector_z_electrolyte,
            shape_func_b_mechanical,
            shape_func_b_times_det_J_b_time_weight_mechanical,
            grad_shape_func_b_x_mechanical,
            grad_shape_func_b_x_times_det_J_b_time_weight_mechanical,
            grad_shape_func_b_y_mechanical,
            grad_shape_func_b_y_times_det_J_b_time_weight_mechanical,
            grad_shape_func_b_z_mechanical,
            grad_shape_func_b_z_times_det_J_b_time_weight_mechanical,
        )
        comp_mechanical_stiffness_matrix = time.time()

        print(
            "time to compute the mechanical stiffness matrix = "
            + "%s seconds" % (comp_mechanical_stiffness_matrix - start_mechanical_time)
        )

        c_G_domain_electrolyte = (
            np.ones(num_gauss_points_in_domain_electrolyte) * c_boundary
        )
        c_G_domain = np.concatenate(
            (c_G_domain_electrolyte, shape_func_electrode * C_new_electrode)
        )

        # compute Beta (expansion coeffieicent)
        beta_fuelcell_expansion_coefficient_electrolyte = np.zeros(
            num_gauss_points_in_domain_electrolyte
        )
        beta_fuelcell_expansion_coefficient_electrode = (
            beta_fuelcell_expansion_coefficient
            * np.ones(num_gauss_points_in_domain_electrode)
        )
        beta_1 = np.concatenate(
            (
                beta_fuelcell_expansion_coefficient_electrolyte,
                beta_fuelcell_expansion_coefficient_electrode,
            )
        ).reshape(num_gauss_points_in_domain_mechanical, 1) * (
            c_G_domain.reshape(num_gauss_points_in_domain_mechanical, 1) - c_boundary
        )
        beta_2 = np.concatenate(
            (
                beta_fuelcell_expansion_coefficient_electrolyte,
                beta_fuelcell_expansion_coefficient_electrode,
            )
        ).reshape(num_gauss_points_in_domain_mechanical, 1) * (
            c_G_domain.reshape(num_gauss_points_in_domain_mechanical, 1) - c_boundary
        )
        beta_3 = np.concatenate(
            (
                beta_fuelcell_expansion_coefficient_electrolyte,
                beta_fuelcell_expansion_coefficient_electrode,
            )
        ).reshape(num_gauss_points_in_domain_mechanical, 1) * (
            c_G_domain.reshape(num_gauss_points_in_domain_mechanical, 1) - c_boundary
        )

        epsilon_D1 = (
            T_c[0, 0].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_1
            + T_c[0, 1].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_2
            + T_c[0, 2].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_3
        )
        epsilon_D2 = (
            T_c[1, 0].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_1
            + T_c[1, 1].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_2
            + T_c[1, 2].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_3
        )
        epsilon_D3 = (
            T_c[2, 0].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_1
            + T_c[2, 1].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_2
            + T_c[2, 2].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_3
        )
        epsilon_D4 = 2 * (
            T_c[3, 0].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_1
            + T_c[3, 1].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_2
            + T_c[3, 2].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_3
        )
        epsilon_D5 = 2 * (
            T_c[4, 0].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_1
            + T_c[4, 1].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_2
            + T_c[4, 2].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_3
        )
        epsilon_D6 = 2 * (
            T_c[5, 0].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_1
            + T_c[5, 1].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_2
            + T_c[5, 2].reshape(num_gauss_points_in_domain_mechanical, 1) * beta_3
        )

        # solve the mechenical part without damage
        f_mechanical = mechanical_force_matrix_3d(
            x_G_mechanical,
            C,
            epsilon_D1,
            epsilon_D2,
            epsilon_D3,
            epsilon_D4,
            epsilon_D5,
            epsilon_D6,
            grad_shape_func_x_times_det_J_time_weight_mechanical,
            grad_shape_func_y_times_det_J_time_weight_mechanical,
            grad_shape_func_z_times_det_J_time_weight_mechanical,
        )

        # if ii==0:
        comp_mechanical_force_matrix = time.time()

        print(
            "time to compute mechanical force matrix= "
            + "%s seconds"
            % (comp_mechanical_force_matrix - comp_mechanical_stiffness_matrix)
        )

        # solve displacement field
        u_disp = spsolve(K_mechanical, f_mechanical)  # 1d array

        ux = u_disp[0:num_nodes_mechanical]  # disp at nodes along x
        uy = u_disp[
            num_nodes_mechanical : 2 * num_nodes_mechanical
        ]  # disp at nodes along y
        uz = u_disp[num_nodes_mechanical * 2 :]  # disp at nodes along y

        ux_gauss = shape_func_mechanical * ux
        uy_gauss = shape_func_mechanical * uy
        uz_gauss = shape_func_mechanical * uz

        # predict the damage factor:
        """
        (grad_shape_func_x*ux) has shape of (number 0f gauss points,), epsilon_D1 has shape of (number of gauss point, 1), if don't reshape, 
        epsilon_x - epsilon_D1 would have shape of (number of gauss point, number of gauss point).
        !!!!! reshape (grad_shape_func_x*ux)
        """

        epsilon_x = (grad_shape_func_x_mechanical * ux).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # normal strain along x at all gauss points
        epsilon_y = (grad_shape_func_y_mechanical * uy).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # normal strain along y at all gauss points
        epsilon_z = (grad_shape_func_z_mechanical * uz).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # normal strain along x at all gauss points

        gamma_xy = (
            (grad_shape_func_x_mechanical * uy + grad_shape_func_y_mechanical * ux)
            * 0.5
        ).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array
        gamma_xz = (
            (grad_shape_func_x_mechanical * uz + grad_shape_func_z_mechanical * ux)
            * 0.5
        ).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array
        gamma_yz = (
            (grad_shape_func_z_mechanical * uy + grad_shape_func_y_mechanical * uz)
            * 0.5
        ).reshape(
            num_gauss_points_in_domain_mechanical, 1
        )  # shear strain aat all gauss points, (grad_shape_func_x*uy+ grad_shape_func_y*ux) is an array

        epsilon_x_mechanical = epsilon_x - epsilon_D1
        epsilon_y_mechanical = epsilon_y - epsilon_D2
        epsilon_z_mechanical = epsilon_z - epsilon_D3
        gamma_xy_mechanical = gamma_xy - epsilon_D4 / 2
        gamma_xz_mechanical = gamma_xz - epsilon_D5 / 2
        gamma_yz_mechanical = gamma_yz - epsilon_D6 / 2

        # calculate the principle strain:
        epsilon_e_eq = (
            2.0
            / 3.0
            * (
                (epsilon_x_mechanical - epsilon_y_mechanical) ** 2
                + (epsilon_x_mechanical - epsilon_z_mechanical) ** 2
                + (epsilon_y_mechanical - epsilon_z_mechanical) ** 2
                + 6
                * (
                    gamma_xy_mechanical**2
                    + gamma_xz_mechanical**2
                    + gamma_yz_mechanical**2
                )
            )
        ) ** 0.5

    k = epsilon_e_eq

    D_damage[np.logical_and(k > k_i, k <= k_f)] = (
        (k[np.logical_and(k > k_i, k <= k_f)] - k_i)
        / (k_f - k_i)
        * k_f
        / k[np.logical_and(k > k_i, k <= k_f)]
    )
    D_damage[k > k_f] = 1.0
    D_damage[k <= k_i] = 0.0

    phi_on_nodes_electrolyte = (
        shape_func_n_nodes_n_nodes_electrolyte * phi_new_electrolyte
    )
    C_on_nodes_electrode = shape_func_n_nodes_n_nodes_electrode * C_new_electrode

    phi_on_GP_electrolyte = shape_func_electrolyte * phi_new_electrolyte
    C_on_GP_electrode = shape_func_electrode * C_new_electrode

    print("on nodes:", np.max(phi_on_nodes_electrolyte))
    print("on nodes:", np.max(C_on_nodes_electrode))

    print("on GP:", np.max(phi_on_GP_electrolyte))
    print("on GP:", np.max(C_on_GP_electrode))

    if dimention == 2:
        potential_on_nodes_save_electrolyte = np.zeros((num_nodes_electrolyte, 3))
        potential_on_nodes_save_electrolyte[:, :2] = x_nodes_electrolyte
        potential_on_nodes_save_electrolyte[:, 2] = phi_on_nodes_electrolyte

        potential_on_GP_save_electrolyte = np.zeros(
            (num_gauss_points_in_domain_electrolyte, 3)
        )
        potential_on_GP_save_electrolyte[:, :2] = x_G_electrolyte
        potential_on_GP_save_electrolyte[:, 2] = (
            shape_func_electrolyte * phi_new_electrolyte
        )

        C_on_nodes_save_electrode = np.zeros((num_nodes_electrode, 3))
        C_on_nodes_save_electrode[:, :2] = x_nodes_electrode
        C_on_nodes_save_electrode[:, 2] = C_on_nodes_electrode

        C_on_GP_save_electrode = np.zeros((num_gauss_points_in_domain_electrode, 3))
        C_on_GP_save_electrode[:, :2] = x_G_electrode
        C_on_GP_save_electrode[:, 2] = shape_func_electrode * C_new_electrode

        # fig3 = plt.figure()
        # plt.scatter(potential_on_nodes_save_electrolyte[:, 0], potential_on_nodes_save_electrolyte[:, 1], c=potential_on_nodes_save_electrolyte[:, 2])
        # plt.colorbar()
        # plt.title('potention in electrolyte (on nodes)')

        # fig4 = plt.figure()
        # plt.scatter(C_on_nodes_save_electrode[:, 0], C_on_nodes_save_electrode[:, 1], c=C_on_nodes_save_electrode[:, 2])
        # plt.colorbar()
        # plt.title('concentration in electrode (on nodes)')

        fig1 = plt.figure()
        plt.scatter(
            potential_on_GP_save_electrolyte[:, 0],
            potential_on_GP_save_electrolyte[:, 1],
            c=potential_on_GP_save_electrolyte[:, 2],
        )
        plt.colorbar()
        plt.title("potention in electrolyte (on GP)")

        fig2 = plt.figure()
        plt.scatter(
            C_on_GP_save_electrode[:, 0],
            C_on_GP_save_electrode[:, 1],
            c=C_on_GP_save_electrode[:, 2],
        )
        plt.colorbar()
        plt.title("concentration in electrode (on GP)")

        fig6 = plt.figure()
        plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=ux_gauss)
        plt.colorbar()
        plt.title("ux")
        # fig61 = plt.figure()
        # plt.scatter(x_G_electrolyte[:, 0], x_G_electrolyte[:, 1], c=ux_gauss[:num_gauss_points_in_domain_electrolyte])
        # plt.colorbar()
        # plt.title('ux')
        # fig62 = plt.figure()
        # plt.scatter(x_G_electrode[:, 0], x_G_electrode[:, 1], c=ux_gauss[num_gauss_points_in_domain_electrolyte:])
        # plt.colorbar()
        # plt.title('ux')

        fig7 = plt.figure()
        plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=uy_gauss)
        plt.colorbar()
        plt.title("uy")
        # fig71 = plt.figure()
        # plt.scatter(x_G_electrolyte[:, 0], x_G_electrolyte[:, 1], c=uy_gauss[:num_gauss_points_in_domain_electrolyte])
        # plt.colorbar()
        # plt.title('uy')
        # fig72 = plt.figure()
        # plt.scatter(x_G_electrode[:, 0], x_G_electrode[:, 1], c=uy_gauss[num_gauss_points_in_domain_electrolyte:])
        # plt.colorbar()
        # plt.title('uy')

        # fig5 = plt.figure()
        # plt.scatter(x_G_mechanical[:, 0], x_G_mechanical[:, 1], c=D_damage)
        # plt.colorbar()
        # plt.title('d')

        plt.show()

    if dimention == 3:

        potential_on_nodes_save_electrolyte = np.zeros((num_nodes_electrolyte, 4))
        potential_on_nodes_save_electrolyte[:, :3] = x_nodes_electrolyte
        potential_on_nodes_save_electrolyte[:, 3] = phi_on_nodes_electrolyte

        potential_on_GP_save_electrolyte = np.zeros(
            (num_gauss_points_in_domain_electrolyte, 4)
        )
        potential_on_GP_save_electrolyte[:, :3] = x_G_electrolyte
        potential_on_GP_save_electrolyte[:, 3] = (
            shape_func_electrolyte * phi_new_electrolyte
        )

        C_on_nodes_save_electrode = np.zeros((num_nodes_electrode, 4))
        C_on_nodes_save_electrode[:, :3] = x_nodes_electrode
        C_on_nodes_save_electrode[:, 3] = C_on_nodes_electrode

        C_on_GP_save_electrode = np.zeros((num_gauss_points_in_domain_electrode, 4))
        C_on_GP_save_electrode[:, :3] = x_G_electrode
        C_on_GP_save_electrode[:, 3] = shape_func_electrode * C_new_electrode

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection="3d")
        sc = ax.scatter(
            potential_on_nodes_save_electrolyte[:, 0],
            potential_on_nodes_save_electrolyte[:, 1],
            potential_on_nodes_save_electrolyte[:, 2],
            c=potential_on_nodes_save_electrolyte[:, 3],
        )
        plt.colorbar(sc, ax=ax)

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection="3d")
        sc = ax.scatter(
            C_on_nodes_save_electrode[:, 0],
            C_on_nodes_save_electrode[:, 1],
            C_on_nodes_save_electrode[:, 2],
            c=C_on_nodes_save_electrode[:, 3],
        )
        plt.colorbar(sc, ax=ax)

        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection="3d")
        sc = ax.scatter(
            potential_on_GP_save_electrolyte[:, 0],
            potential_on_GP_save_electrolyte[:, 1],
            potential_on_GP_save_electrolyte[:, 2],
            c=potential_on_GP_save_electrolyte[:, 3],
        )
        plt.colorbar(sc, ax=ax)

        fig4 = plt.figure()
        ax = fig4.add_subplot(111, projection="3d")
        sc = ax.scatter(
            C_on_GP_save_electrode[:, 0],
            C_on_GP_save_electrode[:, 1],
            C_on_GP_save_electrode[:, 2],
            c=C_on_GP_save_electrode[:, 3],
        )
        plt.colorbar(sc, ax=ax)

        fig5 = plt.figure()
        ax = fig5.add_subplot(111, projection="3d")
        sc = ax.scatter(
            x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=ux_gauss
        )
        plt.colorbar(sc, ax=ax)

        fig6 = plt.figure()
        ax = fig6.add_subplot(111, projection="3d")
        sc = ax.scatter(
            x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=uy_gauss
        )
        plt.colorbar(sc, ax=ax)

        fig7 = plt.figure()
        ax = fig7.add_subplot(111, projection="3d")
        sc = ax.scatter(
            x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=uz_gauss
        )
        plt.colorbar(sc, ax=ax)

        fig8 = plt.figure()
        ax = fig8.add_subplot(111, projection="3d")
        sc = ax.scatter(
            x_G_mechanical[:, 0], x_G_mechanical[:, 1], x_G_mechanical[:, 2], c=D_damage
        )
        plt.colorbar(sc, ax=ax)

        plt.show()

    #     np.savetxt('potential_in_domain_electrolyte.txt', potential_in_domain_save_electrolyte)
    #     np.savetxt('potential_on_boundary_electrolyte.txt', potential_on_boundary_save_electrolyte)

    #     np.savetxt('potential_in_domain_electrode.txt', potential_in_domain_save_electrode)
    #     np.savetxt('potential_on_boundary_electrode.txt', potential_on_boundary_save_electrode)

    #     print(np.max(phi_on_nodes_electrolyte), np.min(phi_on_nodes_electrode))

    # # fig5 = plt.figure()
    # # plt.quiver(potential_in_domain_save_electrolyte[:, 0], potential_in_domain_save_electrolyte[:, 1], flux_x_electrolyte, flux_y_electrolyte, color='b')
    # # plt.xlabel('x')
    # # plt.ylabel('y')
    # # plt.title('Flux Field')

    # # fig6 = plt.figure()
    # # plt.quiver(potential_in_domain_save_electrode[:, 0], potential_in_domain_save_electrode[:, 1], flux_x_electrode, flux_y_electrode, color='b')
    # # plt.xlabel('x')
    # # plt.ylabel('y')
    # # plt.title('Flux Field')
