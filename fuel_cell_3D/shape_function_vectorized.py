"""Vectorized version of compute_phi_M_standard and related functions.

This module provides vectorized implementations that operate on all Gauss points
and nodes simultaneously, eliminating the need for explicit loops.
"""

import numpy as np


def compute_z_and_H_2d_vectorized(
    x_G, x_nodes, a, H_scaling_factor, eps, dtype=np.float64
):
    """Compute z values and H matrices for 2D case (vectorized).

    Args:
        x_G: Array of Gauss points, shape (n_gauss, 2)
        x_nodes: Array of node points, shape (n_nodes, 2)
        a: Array of scaling factors, shape (n_nodes,)
        H_scaling_factor: Scaling factor for H matrices
        eps: Small value to avoid division by zero
        dtype: Data type for arrays (default: np.float32 for memory efficiency)

    Returns:
        Tuple containing vectorized z values, derivatives, and H matrices
    """
    # Convert inputs to specified dtype
    x_G = x_G.astype(dtype, copy=False)
    x_nodes = x_nodes.astype(dtype, copy=False)
    a = a.astype(dtype, copy=False)

    # Expand dimensions for broadcasting
    # x_G: (n_gauss, 1, 2) and x_nodes: (1, n_nodes, 2)
    x_G_exp = x_G[:, np.newaxis, :]  # (n_gauss, 1, 2)
    x_nodes_exp = x_nodes[np.newaxis, :, :]  # (1, n_nodes, 2)

    # Compute differences: (n_gauss, n_nodes, 2)
    diff = x_G_exp - x_nodes_exp

    # Compute distances: (n_gauss, n_nodes)
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # Normalize by a: (n_gauss, n_nodes)
    z = distances / a[np.newaxis, :]

    # Compute partial derivatives
    # Avoid division by zero
    denom = a[np.newaxis, :] * np.maximum(distances, eps)
    z_P_x = diff[:, :, 0] / denom  # (n_gauss, n_nodes)
    z_P_y = diff[:, :, 1] / denom  # (n_gauss, n_nodes)

    # Free memory from distances
    # del distances

    # Compute H matrices
    # H_T shape will be (n_gauss, n_nodes, 3)
    H_T = np.zeros((x_G.shape[0], x_nodes.shape[0], 3), dtype=dtype)
    H_T[:, :, 0] = dtype(1.0)
    H_T[:, :, 1] = diff[:, :, 0] / H_scaling_factor
    H_T[:, :, 2] = diff[:, :, 1] / H_scaling_factor

    # Free memory from diff
    # del diff

    # Derivatives of H_T
    HT_P_x = np.zeros((x_G.shape[0], x_nodes.shape[0], 3), dtype=dtype)
    HT_P_x[:, :, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((x_G.shape[0], x_nodes.shape[0], 3), dtype=dtype)
    HT_P_y[:, :, 2] = dtype(1.0) / H_scaling_factor

    # H is transpose of H_T along last axis
    H = np.transpose(
        H_T, (0, 1, 2)
    )  # Still (n_gauss, n_nodes, 3) but transposed in last dim
    H_P_x = np.transpose(HT_P_x, (0, 1, 2))
    H_P_y = np.transpose(HT_P_y, (0, 1, 2))

    return z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y


def compute_z_and_H_3d_vectorized(
    x_G, x_nodes, a, H_scaling_factor, eps, dtype=np.float64
):
    """Compute z values and H matrices for 3D case (vectorized).

    Args:
        x_G: Array of Gauss points, shape (n_gauss, 3)
        x_nodes: Array of node points, shape (n_nodes, 3)
        a: Array of scaling factors, shape (n_nodes,)
        H_scaling_factor: Scaling factor for H matrices
        eps: Small value to avoid division by zero
        dtype: Data type for arrays (default: np.float32 for memory efficiency)

    Returns:
        Tuple containing vectorized z values, derivatives, and H matrices
    """
    # import pdb
    # pdb.set_trace()

    # Convert inputs to specified dtype
    x_G = x_G.astype(dtype, copy=False)
    x_nodes = x_nodes.astype(dtype, copy=False)
    a = a.astype(dtype, copy=False)

    # Expand dimensions for broadcasting
    x_G_exp = x_G[:, np.newaxis, :]  # (n_gauss, 1, 3)
    x_nodes_exp = x_nodes[np.newaxis, :, :]  # (1, n_nodes, 3)

    # Compute differences: (n_gauss, n_nodes, 3)
    diff = x_G_exp - x_nodes_exp

    # Compute distances: (n_gauss, n_nodes)
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # Normalize by a: (n_gauss, n_nodes)
    z = distances / a[np.newaxis, :]

    # Compute partial derivatives
    denom = a[np.newaxis, :] * np.maximum(distances, eps)

    # save memory
    del distances

    z_P_x = diff[:, :, 0] / denom  # (n_gauss, n_nodes)
    z_P_y = diff[:, :, 1] / denom  # (n_gauss, n_nodes)
    z_P_z = diff[:, :, 2] / denom  # (n_gauss, n_nodes)

    # Compute H matrices
    # H_T shape will be (n_gauss, n_nodes, 4)
    H_T = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    H_T[:, :, 0] = dtype(1.0)
    H_T[:, :, 1] = diff[:, :, 0] / H_scaling_factor
    H_T[:, :, 2] = diff[:, :, 1] / H_scaling_factor
    H_T[:, :, 3] = diff[:, :, 2] / H_scaling_factor

    # save memory
    del diff

    # Derivatives of H_T
    HT_P_x = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    HT_P_x[:, :, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    HT_P_y[:, :, 2] = dtype(1.0) / H_scaling_factor

    HT_P_z = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    HT_P_z[:, :, 3] = dtype(1.0) / H_scaling_factor

    # H is transpose of H_T along last axis
    H = np.transpose(H_T, (0, 1, 2))
    H_P_x = np.transpose(HT_P_x, (0, 1, 2))
    H_P_y = np.transpose(HT_P_y, (0, 1, 2))
    H_P_z = np.transpose(HT_P_z, (0, 1, 2))

    return z, z_P_x, z_P_y, z_P_z, H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z


def compute_phi_kernel(z):
    """Compute phi kernel values based on distance z.

    The kernel is a cubic spline-like function:
    - For 0 <= z < 0.5: phi = 2/3 - 4*z^2 + 4*z^3
    - For 0.5 <= z <= 1: phi = 4/3 - 4*z + 4*z^2 - 4/3*z^3
    - For z > 1: phi = 0

    Args:
        z: Array of normalized distances

    Returns:
        phi: Array of kernel values
        phi_P_z: Array of kernel derivatives w.r.t. z
    """
    phi = np.zeros_like(z)
    phi_P_z = np.zeros_like(z)

    # Mask for different regions
    mask1 = (z >= 0) & (z < 0.5)
    mask2 = (z >= 0.5) & (z <= 1.0)

    # Region 1: 0 <= z < 0.5
    phi[mask1] = 2.0 / 3 - 4 * z[mask1] ** 2 + 4 * z[mask1] ** 3
    phi_P_z[mask1] = -8.0 * z[mask1] + 12.0 * z[mask1] ** 2

    # Region 2: 0.5 <= z <= 1
    phi[mask2] = 4.0 / 3 - 4 * z[mask2] + 4 * z[mask2] ** 2 - 4.0 / 3 * z[mask2] ** 3
    phi_P_z[mask2] = -4 + 8 * z[mask2] - 4 * z[mask2] ** 2

    return phi, phi_P_z


def compute_phi_M_standard_vectorized(
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
    M_P_z=None,
    dtype=np.float64,
):
    """Vectorized version of compute_phi_M_standard.

    This function computes all Gauss-node interactions simultaneously
    using NumPy broadcasting and vectorized operations.

    Args:
        dtype: Data type for computations (default: np.float32 for memory efficiency).
               Use np.float64 for higher precision if needed.
    """

    if M_P_z is None:
        M_P_z = np.zeros_like(M)

    H_scaling_factor = dtype(1.0e-6)
    eps = dtype(2.220446049250313e-16)

    # Determine dimension from M shape
    is_3d = np.shape(M)[1] == 4

    print(f"is_3d: {is_3d}", flush=True)

    # Compute z and H matrices for all point pairs
    if is_3d:
        (
            z,
            z_P_x,
            z_P_y,
            z_P_z,
            H_T,
            HT_P_x,
            HT_P_y,
            HT_P_z,
            H,
            H_P_x,
            H_P_y,
            H_P_z,
        ) = compute_z_and_H_3d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)
    else:
        (z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y) = (
            compute_z_and_H_2d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)
        )
        z_P_z = None
        H_P_z = None
        HT_P_z = None

    # import pdb
    # pdb.set_trace()

    # Compute phi kernel for all distances
    phi, phi_P_z = compute_phi_kernel(z)

    # Find valid pairs (where z <= 1)
    valid_mask = (z >= 0) & (z <= 1.0)

    # Extract indices of valid pairs
    valid_indices = np.where(valid_mask)
    gauss_indices = valid_indices[0]
    node_indices = valid_indices[1]

    # Store nonzero values for sparse matrix construction
    phi_nonzero_index_row = gauss_indices.tolist()
    phi_nonzero_index_column = node_indices.tolist()
    phi_nonzerovalue_data = phi[valid_mask].tolist()

    # Compute phi derivatives
    phi_P_x = phi_P_z * z_P_x
    phi_P_y = phi_P_z * z_P_y
    phi_P_x_nonzerovalue_data = phi_P_x[valid_mask].tolist()
    phi_P_y_nonzerovalue_data = phi_P_y[valid_mask].tolist()

    if is_3d:
        phi_P_z_val = phi_P_z * z_P_z
        phi_P_z_nonzerovalue_data = phi_P_z_val[valid_mask].tolist()
    else:
        phi_P_z_nonzerovalue_data = []

    # Update M matrices using vectorized operations
    # For each Gauss point, accumulate contributions from all valid nodes
    n_gauss = x_G.shape[0]
    n_dim = M.shape[1]

    print(f"Before: Find nodes that contribute to a Gauss pt", flush=True)
    for i in range(n_gauss):
        if i % 10000 == 0:
            print(f"i: {i}")
        # Find nodes that contribute to this Gauss point
        node_mask = valid_mask[i, :]
        if not np.any(node_mask):
            continue

        # Get relevant values for this Gauss point
        phi_i = phi[i, node_mask]
        phi_P_x_i = phi_P_x[i, node_mask]
        phi_P_y_i = phi_P_y[i, node_mask]

        H_i = H[i, node_mask, :]  # (n_valid_nodes, n_dim)
        H_T_i = H_T[i, node_mask, :]  # (n_valid_nodes, n_dim)
        H_P_x_i = H_P_x[i, node_mask, :]
        H_P_y_i = H_P_y[i, node_mask, :]
        HT_P_x_i = HT_P_x[i, node_mask, :]
        HT_P_y_i = HT_P_y[i, node_mask, :]

        # Compute outer products and accumulate
        # Using einsum for efficient tensor operations
        # M[i] += sum over valid nodes of (H * H_T * phi)
        M[i] += np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_i)

        # M_P_x[i] += sum of three terms
        M_P_x[i] += (
            np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_P_x_i)
            + np.einsum("ni,nj,n->ij", H_P_x_i, H_T_i, phi_i)
            + np.einsum("ni,nj,n->ij", H_i, HT_P_x_i, phi_i)
        )

        # M_P_y[i] += sum of three terms
        M_P_y[i] += (
            np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_P_y_i)
            + np.einsum("ni,nj,n->ij", H_P_y_i, H_T_i, phi_i)
            + np.einsum("ni,nj,n->ij", H_i, HT_P_y_i, phi_i)
        )

        if is_3d:
            phi_P_z_i = phi_P_z_val[i, node_mask]
            H_P_z_i = H_P_z[i, node_mask, :]
            HT_P_z_i = HT_P_z[i, node_mask, :]

            M_P_z[i] += (
                np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_P_z_i)
                + np.einsum("ni,nj,n->ij", H_P_z_i, H_T_i, phi_i)
                + np.einsum("ni,nj,n->ij", H_i, HT_P_z_i, phi_i)
            )

    return (
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
    )
