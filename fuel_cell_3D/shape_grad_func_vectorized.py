"""Vectorized version of shape_grad_shape_func and related functions.

This module provides vectorized implementations for computing shape functions
and their gradients, with optimized batch matrix operations.
"""

import numpy as np


def compute_H_matrices_sparse_2d(
    x_G, x_nodes, gauss_indices, node_indices, H_scaling_factor, dtype=np.float32
):
    """Compute H matrices for sparse Gauss-node pairs in 2D.

    Args:
        x_G: Gauss points array
        x_nodes: Node points array
        gauss_indices: Indices of Gauss points in sparse pairs
        node_indices: Indices of nodes in sparse pairs
        H_scaling_factor: Scaling factor for H matrices
        dtype: Data type for computations

    Returns:
        H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y for all sparse pairs
    """
    n_pairs = len(gauss_indices)

    # Get relevant points
    x_G_sparse = x_G[gauss_indices].astype(dtype)
    x_nodes_sparse = x_nodes[node_indices].astype(dtype)

    # Compute differences
    diff = x_G_sparse - x_nodes_sparse  # (n_pairs, 2)

    # Build H_T matrices: (n_pairs, 3)
    H_T = np.zeros((n_pairs, 3), dtype=dtype)
    H_T[:, 0] = dtype(1.0)
    H_T[:, 1] = diff[:, 0] / H_scaling_factor
    H_T[:, 2] = diff[:, 1] / H_scaling_factor

    # Derivatives of H_T
    HT_P_x = np.zeros((n_pairs, 3), dtype=dtype)
    HT_P_x[:, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((n_pairs, 3), dtype=dtype)
    HT_P_y[:, 2] = dtype(1.0) / H_scaling_factor

    # Transpose to get H (just copy since they're 1D in last dimension)
    H = H_T.copy()
    H_P_x = HT_P_x.copy()
    H_P_y = HT_P_y.copy()

    return H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y


def compute_H_matrices_sparse_3d(
    x_G, x_nodes, gauss_indices, node_indices, H_scaling_factor, dtype=np.float32
):
    """Compute H matrices for sparse Gauss-node pairs in 3D.

    Args:
        x_G: Gauss points array
        x_nodes: Node points array
        gauss_indices: Indices of Gauss points in sparse pairs
        node_indices: Indices of nodes in sparse pairs
        H_scaling_factor: Scaling factor for H matrices
        dtype: Data type for computations

    Returns:
        H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z for all sparse pairs
    """
    n_pairs = len(gauss_indices)

    # Get relevant points
    x_G_sparse = x_G[gauss_indices].astype(dtype)
    x_nodes_sparse = x_nodes[node_indices].astype(dtype)

    # Compute differences
    diff = x_G_sparse - x_nodes_sparse  # (n_pairs, 3)

    # Build H_T matrices: (n_pairs, 4)
    H_T = np.zeros((n_pairs, 4), dtype=dtype)
    H_T[:, 0] = dtype(1.0)
    H_T[:, 1] = diff[:, 0] / H_scaling_factor
    H_T[:, 2] = diff[:, 1] / H_scaling_factor
    H_T[:, 3] = diff[:, 2] / H_scaling_factor

    # Derivatives of H_T
    HT_P_x = np.zeros((n_pairs, 4), dtype=dtype)
    HT_P_x[:, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((n_pairs, 4), dtype=dtype)
    HT_P_y[:, 2] = dtype(1.0) / H_scaling_factor

    HT_P_z = np.zeros((n_pairs, 4), dtype=dtype)
    HT_P_z[:, 3] = dtype(1.0) / H_scaling_factor

    # Transpose to get H
    H = H_T.copy()
    H_P_x = HT_P_x.copy()
    H_P_y = HT_P_y.copy()
    H_P_z = HT_P_z.copy()

    return H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z


def batch_matrix_inverse(matrices, dtype=np.float32):
    """Compute inverses of multiple matrices in a batch.

    Args:
        matrices: Array of shape (n_matrices, dim, dim)
        dtype: Data type for computations

    Returns:
        Array of inverted matrices with same shape
    """
    n_matrices = matrices.shape[0]
    dim = matrices.shape[1]

    # Pre-allocate output
    inv_matrices = np.zeros_like(matrices, dtype=dtype)

    # Batch process using vectorized operations where possible
    # For small matrices (3x3 or 4x4), we could use analytical formulas
    # For now, use loop but could optimize further
    for i in range(n_matrices):
        inv_matrices[i] = np.linalg.inv(matrices[i].astype(np.float64)).astype(dtype)

    return inv_matrices


def shape_grad_shape_func_vectorized(
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
    M_P_z=None,
    HT3=None,
    phi_P_z_nonzerovalue_data=None,
    dtype=np.float32,
):
    """Vectorized version of shape_grad_shape_func.

    This function computes shape functions and gradients using vectorized operations
    and batch matrix inversions for improved performance.

    Args:
        dtype: Data type for computations (np.float32 or np.float64)
    """

    # Convert to numpy arrays if needed
    phi_nonzero_index_row = np.array(phi_nonzero_index_row, dtype=np.int32)
    phi_nonzero_index_column = np.array(phi_nonzero_index_column, dtype=np.int32)
    phi_nonzerovalue_data = np.array(phi_nonzerovalue_data, dtype=dtype)

    if phi_P_x_nonzerovalue_data is not None:
        phi_P_x_nonzerovalue_data = np.array(phi_P_x_nonzerovalue_data, dtype=dtype)
        phi_P_y_nonzerovalue_data = np.array(phi_P_y_nonzerovalue_data, dtype=dtype)

    # Determine dimension
    is_3d = M.shape[1] == 4
    n_dim = M.shape[1]

    # Get unique Gauss points that need M inversions
    unique_gauss_points = np.unique(phi_nonzero_index_row)
    n_unique_gauss = len(unique_gauss_points)

    # Batch compute M inverses for all unique Gauss points
    M_subset = M[unique_gauss_points].astype(dtype)
    M_inv_all = batch_matrix_inverse(M_subset, dtype)

    # Create mapping from global to local indices
    gauss_to_local = {g: i for i, g in enumerate(unique_gauss_points)}

    # Pre-compute M_inv for each sparse pair
    M_inv_indices = np.array([gauss_to_local[g] for g in phi_nonzero_index_row])
    M_inv_sparse = M_inv_all[M_inv_indices]  # (num_non_zero_phi_a, n_dim, n_dim)

    # Compute H matrices for all sparse pairs
    H_scaling_factor = dtype(1.0e-6)

    if is_3d:
        (H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z) = (
            compute_H_matrices_sparse_3d(
                x_G,
                x_nodes,
                phi_nonzero_index_row,
                phi_nonzero_index_column,
                H_scaling_factor,
                dtype,
            )
        )
        if phi_P_z_nonzerovalue_data is not None:
            phi_P_z_nonzerovalue_data = np.array(phi_P_z_nonzerovalue_data, dtype=dtype)
    else:
        (H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y) = compute_H_matrices_sparse_2d(
            x_G,
            x_nodes,
            phi_nonzero_index_row,
            phi_nonzero_index_column,
            H_scaling_factor,
            dtype,
        )
        HT_P_z = None
        H_P_z = None

    # Convert HT0, HT1, HT2, HT3 to proper dtype
    HT0 = HT0.astype(dtype)
    if HT1 is not None:
        HT1 = HT1.astype(dtype)
    if HT2 is not None:
        HT2 = HT2.astype(dtype)
    if HT3 is not None and is_3d:
        HT3 = HT3.astype(dtype)

    # Vectorized computation of shape functions
    # shape_func = HT0 @ M_inv @ H * phi
    # H and HT0 are vectors, M_inv is a batch of matrices
    # We need to compute: HT0 @ M_inv[i] @ H[i] for each i
    shape_func_value = (
        np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H) * phi_nonzerovalue_data
    )

    # Initialize gradient arrays
    n_pairs = num_non_zero_phi_a
    grad_shape_func_x_value = np.zeros(n_pairs, dtype=dtype)
    grad_shape_func_y_value = np.zeros(n_pairs, dtype=dtype)
    grad_shape_func_z_value = np.zeros(n_pairs, dtype=dtype) if is_3d else []

    # Compute gradients based on differential method
    if differential_method == "implicite" and IM_RKPM == "False":
        # Implicit method
        grad_shape_func_x_value = (
            np.einsum("j,njk,nk->n", HT1, M_inv_sparse, H) * phi_nonzerovalue_data
        )
        grad_shape_func_y_value = (
            np.einsum("j,njk,nk->n", HT2, M_inv_sparse, H) * phi_nonzerovalue_data
        )

        if is_3d and HT3 is not None:
            grad_shape_func_z_value = (
                np.einsum("j,njk,nk->n", HT3, M_inv_sparse, H) * phi_nonzerovalue_data
            )

    elif differential_method == "direct" or IM_RKPM == "True":
        # Direct method - more complex, involves M_P_x and M_P_y

        # Pre-compute M_P_x and M_P_y for relevant Gauss points
        M_P_x_sparse = M_P_x[phi_nonzero_index_row].astype(dtype)
        M_P_y_sparse = M_P_y[phi_nonzero_index_row].astype(dtype)

        # Compute M_inv_P_x = -M_inv @ M_P_x @ M_inv
        M_inv_P_x = -np.einsum(
            "nij,njk,nkl->nil", M_inv_sparse, M_P_x_sparse, M_inv_sparse
        )
        M_inv_P_y = -np.einsum(
            "nij,njk,nkl->nil", M_inv_sparse, M_P_y_sparse, M_inv_sparse
        )

        # Compute gradient components
        # grad_x = HT0 @ M_inv @ H * phi_P_x + HT0 @ M_inv_P_x @ H * phi + HT0 @ M_inv @ H_P_x * phi
        term1_x = (
            np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H) * phi_P_x_nonzerovalue_data
        )
        term2_x = np.einsum("j,njk,nk->n", HT0, M_inv_P_x, H) * phi_nonzerovalue_data
        term3_x = (
            np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H_P_x) * phi_nonzerovalue_data
        )
        grad_shape_func_x_value = term1_x + term2_x + term3_x

        # Similar for y
        term1_y = (
            np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H) * phi_P_y_nonzerovalue_data
        )
        term2_y = np.einsum("j,njk,nk->n", HT0, M_inv_P_y, H) * phi_nonzerovalue_data
        term3_y = (
            np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H_P_y) * phi_nonzerovalue_data
        )
        grad_shape_func_y_value = term1_y + term2_y + term3_y

        if is_3d and M_P_z is not None:
            M_P_z_sparse = M_P_z[phi_nonzero_index_row].astype(dtype)
            M_inv_P_z = -np.einsum(
                "nij,njk,nkl->nil", M_inv_sparse, M_P_z_sparse, M_inv_sparse
            )

            term1_z = (
                np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H)
                * phi_P_z_nonzerovalue_data
            )
            term2_z = (
                np.einsum("j,njk,nk->n", HT0, M_inv_P_z, H) * phi_nonzerovalue_data
            )
            term3_z = (
                np.einsum("j,njk,nk->n", HT0, M_inv_sparse, H_P_z)
                * phi_nonzerovalue_data
            )
            grad_shape_func_z_value = term1_z + term2_z + term3_z

    # Compute weighted values
    det_J_weights = det_J_time_weight[phi_nonzero_index_row]
    shape_func_times_det_J_time_weight_value = shape_func_value * det_J_weights
    grad_shape_func_x_times_det_J_time_weight_value = (
        grad_shape_func_x_value * det_J_weights
    )
    grad_shape_func_y_times_det_J_time_weight_value = (
        grad_shape_func_y_value * det_J_weights
    )

    if is_3d:
        grad_shape_func_z_times_det_J_time_weight_value = (
            grad_shape_func_z_value * det_J_weights
        )
    else:
        grad_shape_func_z_times_det_J_time_weight_value = []

    # Convert to lists for compatibility
    return (
        shape_func_value.tolist(),
        shape_func_times_det_J_time_weight_value.tolist(),
        grad_shape_func_x_value.tolist(),
        grad_shape_func_y_value.tolist(),
        grad_shape_func_z_value.tolist() if is_3d else [],
        grad_shape_func_x_times_det_J_time_weight_value.tolist(),
        grad_shape_func_y_times_det_J_time_weight_value.tolist(),
        grad_shape_func_z_times_det_J_time_weight_value.tolist() if is_3d else [],
    )
