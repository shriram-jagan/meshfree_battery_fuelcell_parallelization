from typing import Optional, Tuple

from common import csr_array, np


def diffusion_matrix_fuel_cell(
    dimension: int,
    point_or_line_source: np.ndarray,
    shape_func_point_or_line_nodes: csr_array,
    g_diretchlet: np.ndarray,
    beta_Nitsche: np.ndarray,
    normal_vector_x: np.ndarray,
    normal_vector_y: np.ndarray,
    global_diffusion: np.ndarray,
    grad_shape_func_x: csr_array,
    grad_shape_func_y: csr_array,
    grad_shape_func_x_times_det_J_time_weight: csr_array,
    grad_shape_func_y_times_det_J_time_weight: csr_array,
    shape_func_b: csr_array,
    shape_func_b_times_det_J_b_time_weight: csr_array,
    grad_shape_func_b_x_times_det_J_b_time_weight: csr_array,
    grad_shape_func_b_y_times_det_J_b_time_weight: csr_array,
    shape_func_inter_times_det_J_b_time_weight: Optional[csr_array] = None,
    interface_source: Optional[np.ndarray] = None,
    grad_shape_func_z: Optional[csr_array] = None,
    grad_shape_func_z_times_det_J_time_weight: Optional[csr_array] = None,
    grad_shape_func_b_z_times_det_J_b_time_weight: Optional[csr_array] = None,
    normal_vector_z: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    # print('K1')
    K1 = (
        grad_shape_func_x_times_det_J_time_weight.todense() * global_diffusion
    ).T @ grad_shape_func_x.todense() + (
        grad_shape_func_y_times_det_J_time_weight.todense() * global_diffusion
    ).T @ grad_shape_func_y.todense()
    if dimension == 3:
        K1 += (
            grad_shape_func_z_times_det_J_time_weight.todense() * global_diffusion
        ).T @ grad_shape_func_z.todense()

    # print(np.shape(normal_vector_x))
    K2 = (
        -(
            grad_shape_func_b_x_times_det_J_b_time_weight.todense() * normal_vector_x
            + grad_shape_func_b_y_times_det_J_b_time_weight.todense() * normal_vector_y
        ).T
        @ shape_func_b.todense()
    )
    if dimension == 3:
        K2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.todense() * normal_vector_z
        ).T @ shape_func_b.todense()

    # print('K3')
    K3 = (
        shape_func_b.todense() * beta_Nitsche
    ).T @ shape_func_b_times_det_J_b_time_weight.todense()

    K = K1 + K2 + K3

    # print('f1')
    f1 = (
        shape_func_b_times_det_J_b_time_weight.todense() * beta_Nitsche
    ).T @ g_diretchlet

    # print('f2')
    f2 = (
        -(
            grad_shape_func_b_x_times_det_J_b_time_weight.todense() * normal_vector_x
            + grad_shape_func_b_y_times_det_J_b_time_weight.todense() * normal_vector_y
        ).T
        @ g_diretchlet
    )
    if dimension == 3:
        f2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.todense() * normal_vector_z
        ).T @ g_diretchlet

    # when the point source is expressed in delta function times point source value (body source)
    f3 = shape_func_point_or_line_nodes.todense().T @ point_or_line_source

    # interface source (surface source)
    if (
        shape_func_inter_times_det_J_b_time_weight is not None
        and interface_source is not None
    ):
        f4 = -shape_func_inter_times_det_J_b_time_weight.todense().T @ interface_source
    else:
        f4 = np.zeros_like(f1)

    # print('ff')
    f = f1 + f2 + f3 + f4

    return K, f


def diffusion_matrix_fuel_cell_distributed_point_source(
    dimension: int,
    distributed_point_or_line_source: np.ndarray,
    shape_func_distributed_point_or_line_nodes: csr_array,
    g_diretchlet: np.ndarray,
    beta_Nitsche: np.ndarray,
    normal_vector_x: np.ndarray,
    normal_vector_y: np.ndarray,
    global_diffusion: np.ndarray,
    grad_shape_func_x: csr_array,
    grad_shape_func_y: csr_array,
    grad_shape_func_x_times_det_J_time_weight: csr_array,
    grad_shape_func_y_times_det_J_time_weight: csr_array,
    shape_func_b: csr_array,
    shape_func_b_times_det_J_b_time_weight: csr_array,
    grad_shape_func_b_x_times_det_J_b_time_weight: csr_array,
    grad_shape_func_b_y_times_det_J_b_time_weight: csr_array,
    shape_func_inter_times_det_J_b_time_weight: Optional[csr_array] = None,
    interface_source: Optional[np.ndarray] = None,
    grad_shape_func_z: Optional[csr_array] = None,
    grad_shape_func_z_times_det_J_time_weight: Optional[csr_array] = None,
    grad_shape_func_b_z_times_det_J_b_time_weight: Optional[csr_array] = None,
    normal_vector_z: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    # print('K1')
    K1 = (
        grad_shape_func_x_times_det_J_time_weight.todense() * global_diffusion
    ).T @ grad_shape_func_x.todense() + (
        grad_shape_func_y_times_det_J_time_weight.todense() * global_diffusion
    ).T @ grad_shape_func_y.todense()
    if dimension == 3:
        K1 += (
            grad_shape_func_z_times_det_J_time_weight.todense() * global_diffusion
        ).T @ grad_shape_func_z.todense()

    # print(np.shape(normal_vector_x))
    K2 = (
        -(
            grad_shape_func_b_x_times_det_J_b_time_weight.todense() * normal_vector_x
            + grad_shape_func_b_y_times_det_J_b_time_weight.todense() * normal_vector_y
        ).T
        @ shape_func_b.todense()
    )
    if dimension == 3:
        K2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.todense() * normal_vector_z
        ).T @ shape_func_b.todense()

    # print('K3')
    K3 = (
        shape_func_b.todense() * beta_Nitsche
    ).T @ shape_func_b_times_det_J_b_time_weight.todense()

    K = K1 + K2 + K3

    # print('f1')
    f1 = (
        shape_func_b_times_det_J_b_time_weight.todense() * beta_Nitsche
    ).T @ g_diretchlet

    # print('f2')
    f2 = (
        -(
            grad_shape_func_b_x_times_det_J_b_time_weight.todense() * normal_vector_x
            + grad_shape_func_b_y_times_det_J_b_time_weight.todense() * normal_vector_y
        ).T
        @ g_diretchlet
    )
    if dimension == 3:
        f2 -= (
            grad_shape_func_b_z_times_det_J_b_time_weight.todense() * normal_vector_z
        ).T @ g_diretchlet

    # when the point source is expressed in delta function times point source value (surface source)
    f3 = (
        -shape_func_distributed_point_or_line_nodes.todense().T
        @ distributed_point_or_line_source
    )

    # interface source (surface source)
    if (
        shape_func_inter_times_det_J_b_time_weight is not None
        and interface_source is not None
    ):
        f4 = -shape_func_inter_times_det_J_b_time_weight.todense().T @ interface_source
    else:
        f4 = np.zeros_like(f1)

    # print('ff')
    f = f1 + f2 + f3 + f4

    return K, f
