"""
Fully vectorized versions of Gauss point computation functions.
This module provides high-performance implementations using NumPy broadcasting
and vectorization to eliminate loops.
"""

from common import np


def x_G_and_def_J_time_weight_3d_fuelcell_domain_vectorized(
    cell_nodes_x, cell_nodes_y, cell_nodes_z, x_G_domain, weight_G_domain
):
    """
    Fully vectorized computation of Gauss points and Jacobian determinants for 3D hexahedral cells.

    Parameters:
    -----------
    cell_nodes_x : np.ndarray
        Array of shape (n_cells, 8) containing x-coordinates of cell vertices
    cell_nodes_y : np.ndarray
        Array of shape (n_cells, 8) containing y-coordinates of cell vertices
    cell_nodes_z : np.ndarray
        Array of shape (n_cells, 8) containing z-coordinates of cell vertices
    x_G_domain : np.ndarray or list
        Array of shape (n_gauss, 3) containing Gauss points in reference coordinates
    weight_G_domain : np.ndarray or list
        Array of shape (n_gauss,) containing Gauss weights

    Returns:
    --------
    x_G : np.ndarray
        Array of shape (n_cells * n_gauss, 3) containing [x, y, z] coordinates of Gauss points
    det_J_time_weight : np.ndarray
        Array of shape (n_cells * n_gauss,) containing Jacobian determinants times weights
    """

    print(f"begin: x_G_and_def_J_time_weight_3d_fuelcell_domain_vectorized", flush=True)

    # Convert inputs to numpy arrays
    cell_nodes_x = np.array(cell_nodes_x)
    cell_nodes_y = np.array(cell_nodes_y)
    cell_nodes_z = np.array(cell_nodes_z)
    x_G_domain = np.array(x_G_domain)
    weight_G_domain = np.array(weight_G_domain)

    n_cells = cell_nodes_x.shape[0]
    n_gauss = len(x_G_domain)

    if n_cells == 0:
        return np.array([]).reshape(0, 3), np.array([])

    # Extract Gauss point coordinates
    xi = x_G_domain[:, 0]  # ξ coordinates
    eta = x_G_domain[:, 1]  # η coordinates
    zeta = x_G_domain[:, 2]  # ζ coordinates

    # Compute shape functions for all Gauss points
    # Shape: (n_gauss, 8)
    N = np.zeros((n_gauss, 8))
    N[:, 0] = (1 + xi) * (1 - eta) * (1 - zeta) / 8.0
    N[:, 1] = (1 + xi) * (1 + eta) * (1 - zeta) / 8.0
    N[:, 2] = (1 - xi) * (1 + eta) * (1 - zeta) / 8.0
    N[:, 3] = (1 - xi) * (1 - eta) * (1 - zeta) / 8.0
    N[:, 4] = (1 + xi) * (1 - eta) * (1 + zeta) / 8.0
    N[:, 5] = (1 + xi) * (1 + eta) * (1 + zeta) / 8.0
    N[:, 6] = (1 - xi) * (1 + eta) * (1 + zeta) / 8.0
    N[:, 7] = (1 - xi) * (1 - eta) * (1 + zeta) / 8.0

    # Compute Gauss points for all cells and all Gauss points
    # Using Einstein summation: x_G[cell, gauss] = sum over nodes of N[gauss, node] * x[cell, node]
    x_G_all = np.einsum("gi,ci->cg", N, cell_nodes_x)  # Shape: (n_cells, n_gauss)
    y_G_all = np.einsum("gi,ci->cg", N, cell_nodes_y)
    z_G_all = np.einsum("gi,ci->cg", N, cell_nodes_z)

    # Stack and reshape to (n_cells * n_gauss, 3)
    x_G = np.stack([x_G_all.ravel(), y_G_all.ravel(), z_G_all.ravel()], axis=1)

    # Compute shape function derivatives
    # dN/dξ, dN/dη, dN/dζ for all Gauss points
    dN_dxi = np.zeros((n_gauss, 8))
    dN_deta = np.zeros((n_gauss, 8))
    dN_dzeta = np.zeros((n_gauss, 8))

    # Derivatives with respect to ξ
    dN_dxi[:, 0] = (1 - eta) * (1 - zeta) / 8.0
    dN_dxi[:, 1] = (1 + eta) * (1 - zeta) / 8.0
    dN_dxi[:, 2] = -(1 + eta) * (1 - zeta) / 8.0
    dN_dxi[:, 3] = -(1 - eta) * (1 - zeta) / 8.0
    dN_dxi[:, 4] = (1 - eta) * (1 + zeta) / 8.0
    dN_dxi[:, 5] = (1 + eta) * (1 + zeta) / 8.0
    dN_dxi[:, 6] = -(1 + eta) * (1 + zeta) / 8.0
    dN_dxi[:, 7] = -(1 - eta) * (1 + zeta) / 8.0

    # Derivatives with respect to η
    dN_deta[:, 0] = -(1 + xi) * (1 - zeta) / 8.0
    dN_deta[:, 1] = (1 + xi) * (1 - zeta) / 8.0
    dN_deta[:, 2] = (1 - xi) * (1 - zeta) / 8.0
    dN_deta[:, 3] = -(1 - xi) * (1 - zeta) / 8.0
    dN_deta[:, 4] = -(1 + xi) * (1 + zeta) / 8.0
    dN_deta[:, 5] = (1 + xi) * (1 + zeta) / 8.0
    dN_deta[:, 6] = (1 - xi) * (1 + zeta) / 8.0
    dN_deta[:, 7] = -(1 - xi) * (1 + zeta) / 8.0

    # Derivatives with respect to ζ
    dN_dzeta[:, 0] = -(1 + xi) * (1 - eta) / 8.0
    dN_dzeta[:, 1] = -(1 + xi) * (1 + eta) / 8.0
    dN_dzeta[:, 2] = -(1 - xi) * (1 + eta) / 8.0
    dN_dzeta[:, 3] = -(1 - xi) * (1 - eta) / 8.0
    dN_dzeta[:, 4] = (1 + xi) * (1 - eta) / 8.0
    dN_dzeta[:, 5] = (1 + xi) * (1 + eta) / 8.0
    dN_dzeta[:, 6] = (1 - xi) * (1 + eta) / 8.0
    dN_dzeta[:, 7] = (1 - xi) * (1 - eta) / 8.0

    # Compute Jacobian components for all cells and all Gauss points
    # J[i,j] = dx_i/dξ_j where x_i = (x,y,z) and ξ_j = (ξ,η,ζ)

    # J11 = dx/dξ for all cells and Gauss points
    J11 = np.einsum("gi,ci->cg", dN_dxi, cell_nodes_x)
    J12 = np.einsum("gi,ci->cg", dN_deta, cell_nodes_x)
    J13 = np.einsum("gi,ci->cg", dN_dzeta, cell_nodes_x)

    # J21 = dy/dξ for all cells and Gauss points
    J21 = np.einsum("gi,ci->cg", dN_dxi, cell_nodes_y)
    J22 = np.einsum("gi,ci->cg", dN_deta, cell_nodes_y)
    J23 = np.einsum("gi,ci->cg", dN_dzeta, cell_nodes_y)

    # J31 = dz/dξ for all cells and Gauss points
    J31 = np.einsum("gi,ci->cg", dN_dxi, cell_nodes_z)
    J32 = np.einsum("gi,ci->cg", dN_deta, cell_nodes_z)
    J33 = np.einsum("gi,ci->cg", dN_dzeta, cell_nodes_z)

    # Compute determinant of Jacobian for all cells and Gauss points
    # det(J) = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31)
    det_J = (
        J11 * (J22 * J33 - J23 * J32)
        - J12 * (J21 * J33 - J23 * J31)
        + J13 * (J21 * J32 - J22 * J31)
    )

    # Multiply by weights (broadcasting weights to all cells)
    det_J_time_weight = det_J * weight_G_domain[np.newaxis, :]

    # Flatten to 1D array
    det_J_time_weight = det_J_time_weight.ravel()

    print(f"  end: x_G_and_def_J_time_weight_3d_fuelcell_domain_vectorized", flush=True)

    return x_G, det_J_time_weight


def x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface_vectorized(
    cell_nodes_boundary_x,
    cell_nodes_boundary_y,
    cell_nodes_boundary_z,
    x_G_domain,
    weight_G_domain,
):
    """
    Fully vectorized computation of Gauss points and Jacobian determinants for 2D boundary interfaces
    in a 3D fuel cell domain. This version processes all cells of each type simultaneously.

    Parameters:
    -----------
    cell_nodes_boundary_x/y/z : np.ndarray
        Arrays of shape (n_cells, 4) containing coordinates of vertices for each boundary cell
    x_G_domain : np.ndarray
        Array of shape (n_gauss, 2) containing reference Gauss quadrature points
    weight_G_domain : np.ndarray
        Array of shape (n_gauss,) containing Gauss weights

    Returns:
    --------
    x_G : list
        List of [x, y, z] coordinates for all Gauss points
    det_J_time_weight : list
        List of Jacobian determinants multiplied by weights
    """

    print(
        f"begin: x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface_vectorized",
        flush=True,
    )

    # Convert inputs to numpy arrays if needed
    cell_nodes_boundary_x = np.array(cell_nodes_boundary_x)
    cell_nodes_boundary_y = np.array(cell_nodes_boundary_y)
    cell_nodes_boundary_z = np.array(cell_nodes_boundary_z)
    x_G_domain = np.array(x_G_domain)
    weight_G_domain = np.array(weight_G_domain)

    if cell_nodes_boundary_x.shape[0] == 0:
        return [], []

    n_cells = cell_nodes_boundary_x.shape[0]
    n_gauss = len(x_G_domain)

    # Check for constant coordinates (with tolerance for numerical precision)
    tol = 1e-10

    # Check which cells have constant y, x, or z
    y_constant = (
        np.abs(cell_nodes_boundary_y.max(axis=1) - cell_nodes_boundary_y.min(axis=1))
        < tol
    )
    x_constant = (
        np.abs(cell_nodes_boundary_x.max(axis=1) - cell_nodes_boundary_x.min(axis=1))
        < tol
    )
    z_constant = (
        np.abs(cell_nodes_boundary_z.max(axis=1) - cell_nodes_boundary_z.min(axis=1))
        < tol
    )

    x_G = []
    det_J_time_weight = []

    # Prepare shape functions for bilinear interpolation
    xi = x_G_domain[:, 0]  # First coordinate in reference space
    eta = x_G_domain[:, 1]  # Second coordinate in reference space

    # Shape functions at Gauss points (n_gauss, 4)
    N = np.zeros((n_gauss, 4))
    N[:, 0] = 0.25 * (1 - xi) * (1 - eta)
    N[:, 1] = 0.25 * (1 + xi) * (1 - eta)
    N[:, 2] = 0.25 * (1 + xi) * (1 + eta)
    N[:, 3] = 0.25 * (1 - xi) * (1 + eta)

    # Derivatives of shape functions with respect to xi (n_gauss, 4)
    dN_dxi = np.zeros((n_gauss, 4))
    dN_dxi[:, 0] = -0.25 * (1 - eta)
    dN_dxi[:, 1] = 0.25 * (1 - eta)
    dN_dxi[:, 2] = 0.25 * (1 + eta)
    dN_dxi[:, 3] = -0.25 * (1 + eta)

    # Derivatives of shape functions with respect to eta (n_gauss, 4)
    dN_deta = np.zeros((n_gauss, 4))
    dN_deta[:, 0] = -0.25 * (1 - xi)
    dN_deta[:, 1] = -0.25 * (1 + xi)
    dN_deta[:, 2] = 0.25 * (1 + xi)
    dN_deta[:, 3] = 0.25 * (1 - xi)

    # Process y-constant surfaces (perpendicular to y-axis) - fully vectorized
    if np.any(y_constant):
        y_const_idx = np.where(y_constant)[0]
        n_y_cells = len(y_const_idx)

        # Get vertices for all y-constant cells
        x_ver = cell_nodes_boundary_x[y_const_idx, :]  # (n_y_cells, 4)
        y_ver = cell_nodes_boundary_y[y_const_idx, :]  # (n_y_cells, 4)
        z_ver = cell_nodes_boundary_z[y_const_idx, :]  # (n_y_cells, 4)

        # Compute physical coordinates for all cells and all Gauss points
        # Using einsum for efficient batch matrix multiplication
        # Result shape: (n_y_cells, n_gauss)
        x_G_points = np.einsum("gj,ij->ig", N, x_ver)  # (n_y_cells, n_gauss)
        z_G_points = np.einsum("gj,ij->ig", N, z_ver)  # (n_y_cells, n_gauss)
        y_G_points = np.repeat(y_ver[:, 0:1], n_gauss, axis=1)  # (n_y_cells, n_gauss)

        # Compute Jacobian components for all cells and Gauss points
        J11 = np.einsum("gj,ij->ig", dN_dxi, x_ver)  # dx/dxi
        J12 = np.einsum("gj,ij->ig", dN_dxi, z_ver)  # dz/dxi
        J21 = np.einsum("gj,ij->ig", dN_deta, x_ver)  # dx/deta
        J22 = np.einsum("gj,ij->ig", dN_deta, z_ver)  # dz/deta

        # Compute determinants
        det_J = J11 * J22 - J12 * J21  # (n_y_cells, n_gauss)

        # Multiply by weights
        det_J_weighted = det_J * weight_G_domain[np.newaxis, :]  # (n_y_cells, n_gauss)

        # Flatten and add to results
        for i in range(n_y_cells):
            for k in range(n_gauss):
                x_G.append([x_G_points[i, k], y_G_points[i, k], z_G_points[i, k]])
                det_J_time_weight.append(det_J_weighted[i, k])

    # Process x-constant surfaces (perpendicular to x-axis) - fully vectorized
    if np.any(x_constant):
        x_const_idx = np.where(x_constant)[0]
        n_x_cells = len(x_const_idx)

        # Get vertices for all x-constant cells
        x_ver = cell_nodes_boundary_x[x_const_idx, :]  # (n_x_cells, 4)
        y_ver = cell_nodes_boundary_y[x_const_idx, :]  # (n_x_cells, 4)
        z_ver = cell_nodes_boundary_z[x_const_idx, :]  # (n_x_cells, 4)

        # Compute physical coordinates for all cells and all Gauss points
        y_G_points = np.einsum("gj,ij->ig", N, y_ver)  # (n_x_cells, n_gauss)
        z_G_points = np.einsum("gj,ij->ig", N, z_ver)  # (n_x_cells, n_gauss)
        x_G_points = np.repeat(x_ver[:, 0:1], n_gauss, axis=1)  # (n_x_cells, n_gauss)

        # Compute Jacobian components for all cells and Gauss points
        J11 = np.einsum("gj,ij->ig", dN_dxi, y_ver)  # dy/dxi
        J12 = np.einsum("gj,ij->ig", dN_dxi, z_ver)  # dz/dxi
        J21 = np.einsum("gj,ij->ig", dN_deta, y_ver)  # dy/deta
        J22 = np.einsum("gj,ij->ig", dN_deta, z_ver)  # dz/deta

        # Compute determinants
        det_J = J11 * J22 - J12 * J21  # (n_x_cells, n_gauss)

        # Multiply by weights
        det_J_weighted = det_J * weight_G_domain[np.newaxis, :]  # (n_x_cells, n_gauss)

        # Flatten and add to results
        for i in range(n_x_cells):
            for k in range(n_gauss):
                x_G.append([x_G_points[i, k], y_G_points[i, k], z_G_points[i, k]])
                det_J_time_weight.append(det_J_weighted[i, k])

    # Process z-constant surfaces (perpendicular to z-axis) - fully vectorized
    if np.any(z_constant):
        z_const_idx = np.where(z_constant)[0]
        n_z_cells = len(z_const_idx)

        # Get vertices for all z-constant cells
        x_ver = cell_nodes_boundary_x[z_const_idx, :]  # (n_z_cells, 4)
        y_ver = cell_nodes_boundary_y[z_const_idx, :]  # (n_z_cells, 4)
        z_ver = cell_nodes_boundary_z[z_const_idx, :]  # (n_z_cells, 4)

        # Compute physical coordinates for all cells and all Gauss points
        x_G_points = np.einsum("gj,ij->ig", N, x_ver)  # (n_z_cells, n_gauss)
        y_G_points = np.einsum("gj,ij->ig", N, y_ver)  # (n_z_cells, n_gauss)
        z_G_points = np.repeat(z_ver[:, 0:1], n_gauss, axis=1)  # (n_z_cells, n_gauss)

        # Compute Jacobian components for all cells and Gauss points
        J11 = np.einsum("gj,ij->ig", dN_dxi, x_ver)  # dx/dxi
        J12 = np.einsum("gj,ij->ig", dN_dxi, y_ver)  # dy/dxi
        J21 = np.einsum("gj,ij->ig", dN_deta, x_ver)  # dx/deta
        J22 = np.einsum("gj,ij->ig", dN_deta, y_ver)  # dy/deta

        # Compute determinants
        det_J = J11 * J22 - J12 * J21  # (n_z_cells, n_gauss)

        # Multiply by weights
        det_J_weighted = det_J * weight_G_domain[np.newaxis, :]  # (n_z_cells, n_gauss)

        # Flatten and add to results
        for i in range(n_z_cells):
            for k in range(n_gauss):
                x_G.append([x_G_points[i, k], y_G_points[i, k], z_G_points[i, k]])
                det_J_time_weight.append(det_J_weighted[i, k])

    print(
        f"  end: x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface_vectorized",
        flush=True,
    )

    return x_G, det_J_time_weight


def x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_vectorized(
    cell_nodes_boundary_x,
    cell_nodes_boundary_z,
    y_coords_on_boundary,
    x_G_domain,
    weight_G_domain,
):
    """
    Fully vectorized computation of Gauss points and Jacobian determinants for 2D boundaries
    in a 3D fuel cell domain with fixed y-coordinate.

    Parameters:
    -----------
    cell_nodes_boundary_x : np.ndarray
        Array of shape (n_cells, 4) containing x-coordinates of boundary cell vertices
    cell_nodes_boundary_z : np.ndarray
        Array of shape (n_cells, 4) containing z-coordinates of boundary cell vertices
    y_coords_on_boundary : float
        Fixed y-coordinate value for all points on this boundary
    x_G_domain : np.ndarray
        Array of shape (n_gauss, 2) containing reference Gauss quadrature points
    weight_G_domain : np.ndarray
        Array of shape (n_gauss,) containing Gauss weights

    Returns:
    --------
    x_G : list
        List of [x, y, z] coordinates for all Gauss points
    det_J_time_weight : list
        List of Jacobian determinants multiplied by weights
    """

    print(
        f"begin: x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_vectorized",
        flush=True,
    )

    # Convert inputs to numpy arrays
    cell_nodes_boundary_x = np.array(cell_nodes_boundary_x)
    cell_nodes_boundary_z = np.array(cell_nodes_boundary_z)
    x_G_domain = np.array(x_G_domain)
    weight_G_domain = np.array(weight_G_domain)

    if cell_nodes_boundary_x.shape[0] == 0:
        return [], []

    n_cells = cell_nodes_boundary_x.shape[0]
    n_gauss = len(x_G_domain)

    # Extract reference coordinates
    xi = x_G_domain[:, 0]  # ξ coordinates
    eta = x_G_domain[:, 1]  # η coordinates

    # Compute 2D shape functions for quadrilateral elements
    # Shape: (n_gauss, 4)
    N = np.zeros((n_gauss, 4))
    N[:, 0] = 0.25 * (1 - xi) * (1 - eta)
    N[:, 1] = 0.25 * (1 + xi) * (1 - eta)
    N[:, 2] = 0.25 * (1 + xi) * (1 + eta)
    N[:, 3] = 0.25 * (1 - xi) * (1 + eta)

    # Derivatives of shape functions with respect to xi
    # Shape: (n_gauss, 4)
    dN_dxi = np.zeros((n_gauss, 4))
    dN_dxi[:, 0] = -0.25 * (1 - eta)
    dN_dxi[:, 1] = 0.25 * (1 - eta)
    dN_dxi[:, 2] = 0.25 * (1 + eta)
    dN_dxi[:, 3] = -0.25 * (1 + eta)

    # Derivatives of shape functions with respect to eta
    # Shape: (n_gauss, 4)
    dN_deta = np.zeros((n_gauss, 4))
    dN_deta[:, 0] = -0.25 * (1 - xi)
    dN_deta[:, 1] = -0.25 * (1 + xi)
    dN_deta[:, 2] = 0.25 * (1 + xi)
    dN_deta[:, 3] = 0.25 * (1 - xi)

    # Compute physical coordinates for all cells and all Gauss points
    # Using Einstein summation: x_G[cell, gauss] = sum over nodes of N[gauss, node] * x[cell, node]
    x_G_all = np.einsum(
        "gi,ci->cg", N, cell_nodes_boundary_x
    )  # Shape: (n_cells, n_gauss)
    z_G_all = np.einsum(
        "gi,ci->cg", N, cell_nodes_boundary_z
    )  # Shape: (n_cells, n_gauss)

    # y-coordinate is constant for all points
    y_G_all = np.full((n_cells, n_gauss), y_coords_on_boundary)

    # Compute Jacobian components for all cells and Gauss points
    # J = [[dx/dxi, dz/dxi], [dx/deta, dz/deta]]
    J11 = np.einsum("gi,ci->cg", dN_dxi, cell_nodes_boundary_x)  # dx/dxi
    J12 = np.einsum("gi,ci->cg", dN_dxi, cell_nodes_boundary_z)  # dz/dxi
    J21 = np.einsum("gi,ci->cg", dN_deta, cell_nodes_boundary_x)  # dx/deta
    J22 = np.einsum("gi,ci->cg", dN_deta, cell_nodes_boundary_z)  # dz/deta

    # Compute determinant of Jacobian for all cells and Gauss points
    det_J = J11 * J22 - J12 * J21  # Shape: (n_cells, n_gauss)

    # Multiply by weights
    det_J_weighted = det_J * weight_G_domain[np.newaxis, :]  # Shape: (n_cells, n_gauss)

    # Flatten and create output lists
    x_G = []
    det_J_time_weight = []

    for i in range(n_cells):
        for k in range(n_gauss):
            x_G.append([x_G_all[i, k], y_G_all[i, k], z_G_all[i, k]])
            det_J_time_weight.append(det_J_weighted[i, k])

    print(
        f"  end: x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_vectorized",
        flush=True,
    )

    return x_G, det_J_time_weight
