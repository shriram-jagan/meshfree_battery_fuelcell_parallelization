"""
Diffusion matrix assembly module using Nitsche's method.

This module assembles the stiffness matrix and force vector for the
steady-state diffusion equation with Dirichlet boundary conditions
enforced using Nitsche's method (weak enforcement).

The discretized system is: K @ c = f

where K = K1 + K2 + K3:
    - K1: Domain diffusion integral
    - K2: Flux consistency term on boundary
    - K3: Nitsche penalty term on boundary
"""

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
    """
    Assemble diffusion stiffness matrix and force vector using Nitsche's method.

    This function constructs the linear system for the steady-state diffusion
    equation with Dirichlet boundary conditions enforced weakly via Nitsche's
    method. Used for delta function (point) source terms.

    Parameters
    ----------
    dimension : int
        Problem dimension (2 or 3).
    point_or_line_source : np.ndarray
        Source term values at source points, shape (n_source,).
    shape_func_point_or_line_nodes : csr_array
        Shape functions at source points, shape (n_source, n_nodes).
    g_diretchlet : np.ndarray
        Dirichlet boundary values, shape (n_gauss_boundary,).
    beta_Nitsche : np.ndarray
        Nitsche penalty parameter, shape (n_gauss_boundary,).
    normal_vector_x, normal_vector_y : np.ndarray
        Outward normal components at boundary Gauss points.
    global_diffusion : np.ndarray
        Diffusion coefficient at domain Gauss points.
    grad_shape_func_x, grad_shape_func_y : csr_array
        Shape function gradients, shape (n_gauss, n_nodes).
    grad_shape_func_x_times_det_J_time_weight : csr_array
        Weighted gradient matrices for integration, shape (n_gauss, n_nodes).
    shape_func_b : csr_array
        Shape functions at boundary Gauss points.
    shape_func_b_times_det_J_b_time_weight : csr_array
        Weighted boundary shape functions.
    grad_shape_func_b_*_times_det_J_b_time_weight : csr_array
        Weighted boundary gradient matrices.
    shape_func_inter_times_det_J_b_time_weight : csr_array, optional
        Shape functions at interface for interface source.
    interface_source : np.ndarray, optional
        Interface source term values.
    grad_shape_func_z : csr_array, optional
        z-gradient of shape functions (3D only).
    normal_vector_z : np.ndarray, optional
        z-component of outward normal (3D only).

    Returns
    -------
    K : np.ndarray
        Stiffness matrix, shape (n_nodes, n_nodes).
    f : np.ndarray
        Force vector, shape (n_nodes,).
    """

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
