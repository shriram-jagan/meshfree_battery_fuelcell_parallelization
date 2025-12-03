import numpy as np


def diffusion_matrix_fuel_cell(
    dimension,
    point_or_line_source,
    shape_func_point_or_line_nodes,
    g_diretchlet,
    beta_Nitsche,
    normal_vector_x,
    normal_vector_y,
    global_diffusion,
    grad_shape_func_x,
    grad_shape_func_y,
    grad_shape_func_x_times_det_J_time_weight,
    grad_shape_func_y_times_det_J_time_weight,
    shape_func_b,
    shape_func_b_times_det_J_b_time_weight,
    grad_shape_func_b_x_times_det_J_b_time_weight,
    grad_shape_func_b_y_times_det_J_b_time_weight,
    shape_func_inter_times_det_J_b_time_weight=None,
    interface_source=None,
    grad_shape_func_z=None,
    grad_shape_func_z_times_det_J_time_weight=None,
    grad_shape_func_b_z_times_det_J_b_time_weight=None,
    normal_vector_z=None,
):

    # print('K1')
    K1 = (
        (grad_shape_func_x_times_det_J_time_weight).multiply(global_diffusion)
    ).T @ grad_shape_func_x + (
        (grad_shape_func_y_times_det_J_time_weight).multiply(global_diffusion)
    ).T @ grad_shape_func_y
    if dimension == 3:
        K1 += (
            (grad_shape_func_z_times_det_J_time_weight).multiply(global_diffusion)
        ).T @ grad_shape_func_z

    # print(np.shape(normal_vector_x))
    K2 = (
        -(
            (grad_shape_func_b_x_times_det_J_b_time_weight).multiply(normal_vector_x)
            + grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)
        ).T
        @ shape_func_b
    )
    if dimension == 3:
        K2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)
        ).T @ shape_func_b

    # print('K3')
    K3 = (
        shape_func_b.multiply(beta_Nitsche)
    ).T @ shape_func_b_times_det_J_b_time_weight

    K = K1 + K2 + K3

    # print('f1')
    f1 = (
        (shape_func_b_times_det_J_b_time_weight.multiply(beta_Nitsche)).T
    ) @ g_diretchlet

    # print('f2')
    f2 = (
        -(
            grad_shape_func_b_x_times_det_J_b_time_weight.multiply(normal_vector_x)
            + grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)
        ).T
        @ g_diretchlet
    )
    if dimension == 3:
        f2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)
        ).T @ g_diretchlet

    # when the point source is expressed in delta function times point source value (body source)
    f3 = shape_func_point_or_line_nodes.T @ point_or_line_source

    # interface source (surface source)
    f4 = -shape_func_inter_times_det_J_b_time_weight.T @ interface_source

    # print('ff')
    f = f1 + f2 + f3 + f4

    return K, f


def diffusion_matrix_fuel_cell_distributed_point_source(
    dimension,
    distributed_point_or_line_source,
    shape_func_distributed_point_or_line_nodes,
    g_diretchlet,
    beta_Nitsche,
    normal_vector_x,
    normal_vector_y,
    global_diffusion,
    grad_shape_func_x,
    grad_shape_func_y,
    grad_shape_func_x_times_det_J_time_weight,
    grad_shape_func_y_times_det_J_time_weight,
    shape_func_b,
    shape_func_b_times_det_J_b_time_weight,
    grad_shape_func_b_x_times_det_J_b_time_weight,
    grad_shape_func_b_y_times_det_J_b_time_weight,
    shape_func_inter_times_det_J_b_time_weight=None,
    interface_source=None,
    grad_shape_func_z=None,
    grad_shape_func_z_times_det_J_time_weight=None,
    grad_shape_func_b_z_times_det_J_b_time_weight=None,
    normal_vector_z=None,
):

    # print('K1')
    K1 = (
        (grad_shape_func_x_times_det_J_time_weight).multiply(global_diffusion)
    ).T @ grad_shape_func_x + (
        (grad_shape_func_y_times_det_J_time_weight).multiply(global_diffusion)
    ).T @ grad_shape_func_y
    if dimension == 3:
        K1 += (
            (grad_shape_func_z_times_det_J_time_weight).multiply(global_diffusion)
        ).T @ grad_shape_func_z

    # print(np.shape(normal_vector_x))
    K2 = (
        -(
            (grad_shape_func_b_x_times_det_J_b_time_weight).multiply(normal_vector_x)
            + grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)
        ).T
        @ shape_func_b
    )
    if dimension == 3:
        K2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)
        ).T @ shape_func_b

    # print('K3')
    K3 = (
        shape_func_b.multiply(beta_Nitsche)
    ).T @ shape_func_b_times_det_J_b_time_weight

    K = K1 + K2 + K3

    # print('f1')
    f1 = (
        (shape_func_b_times_det_J_b_time_weight.multiply(beta_Nitsche)).T
    ) @ g_diretchlet

    # print('f2')
    f2 = (
        -(
            grad_shape_func_b_x_times_det_J_b_time_weight.multiply(normal_vector_x)
            + grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)
        ).T
        @ g_diretchlet
    )
    if dimension == 3:
        f2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)
        ).T @ g_diretchlet

    # when the point source is expressed in delta function times point source value (surface slource)
    f3 = (
        -shape_func_distributed_point_or_line_nodes.T @ distributed_point_or_line_source
    )

    # interface source (surface source)
    f4 = -shape_func_inter_times_det_J_b_time_weight.T @ interface_source

    # print('ff')
    f = f1 + f2 + f3 + f4

    return K, f
