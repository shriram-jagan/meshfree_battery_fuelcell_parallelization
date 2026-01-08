"""
RKPM shape function computation module.

This module computes Reproducing Kernel Particle Method (RKPM) shape functions
and their derivatives. The RKPM shape function for node I at point x is:

    Ψ_I(x) = H^T(0) M^{-1}(x) H(x - x_I) φ(z)

where:
    - H(x) is the polynomial basis vector [1, x/h, y/h, z/h]^T
    - M(x) is the moment matrix (ensures reproduction of polynomials)
    - φ(z) is the cubic B-spline kernel function
    - z = ||x - x_I|| / a_I is the normalized distance

The module provides both original loop-based and optimized vectorized
implementations for performance.
"""

from common import np

# Try to import the vectorized versions if available
try:
    from shape_function_vectorized import (
        compute_phi_M_standard_sparse,
        compute_phi_M_standard_vectorized,
    )

    VECTORIZED_PHI_AVAILABLE = True
    SPARSE_PHI_AVAILABLE = True
except ImportError:
    VECTORIZED_PHI_AVAILABLE = False
    SPARSE_PHI_AVAILABLE = False

try:
    from shape_grad_func_vectorized import shape_grad_shape_func_vectorized

    VECTORIZED_GRAD_AVAILABLE = True
except ImportError:
    VECTORIZED_GRAD_AVAILABLE = False

# Keep old name for compatibility
VECTORIZED_AVAILABLE = VECTORIZED_PHI_AVAILABLE


def compute_phi_M_with_interface_method(
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
):
    """Compute phi_M using interface method (IM_RKPM=True and single_grain=False)."""

    print(f"Compute phi_M using interface method", flush=True)

    if M_P_z is None:
        # Initialize with zeros or correct shape
        M_P_z = np.zeros_like(M)  # Adjust shape accordingly

    phi_nonzero_index_row = []
    phi_nonzero_index_column = []
    phi_nonzerovalue_data = []
    phi_P_x_nonzerovalue_data = []
    phi_P_y_nonzerovalue_data = []
    phi_P_z_nonzerovalue_data = []
    z = []
    z_P_x = []
    z_P_y = []
    z_P_z = []
    phipz = []

    print("get in", flush=True)
    for i in range(np.shape(x_G)[0]):
        """
        check the distance between point and segments, exact distance
        """
        dx_distance = np.zeros(num_interface_segments)

        # if x_nodes[j,:] not in interface_nodes:
        # find the minimum distance of gauss point to interface
        # if gauss point is A, boundary segment is BC, if (BA dot BC) times (CA dot CB) is negative, it means the vertical line from A to segment BC intersect BC on its extension,
        # in this case, the distance from A to BC is min(|AB|, |AC|)
        # if (BA dot BC) times (CA dot CB) is positive, there is a point D with BC that AD is normal to BC. the distance from A to BC is |BA vector - [vector BA dot (BC vector divided by |BC|) times (BC vector divided by |BC|) ]|
        BA = x_G[i, :] - BxByCxCy[:, :2]
        BC = BxByCxCy[:, 2:4] - BxByCxCy[:, :2]
        CB = -BxByCxCy[:, 2:4] + BxByCxCy[:, :2]
        CA = x_G[i, :] - BxByCxCy[:, 2:4]

        BA_dot_BC = BA[:, 0] * BC[:, 0] + BA[:, 1] * BC[:, 1]
        CA_dot_CB = CA[:, 0] * CB[:, 0] + CA[:, 1] * CB[:, 1]

        sign_extension = BA_dot_BC * CA_dot_CB

        positive_index = np.where(sign_extension > 0)[0]
        negative_index = np.where(sign_extension < 0)[0]
        zero_index = np.where(sign_extension == 0)[0]

        BA_dot_unit_BC = BA_dot_BC / (((BC[:, 0]) ** 2 + (BC[:, 1]) ** 2) ** 0.5)

        BA_dot_unit_BC_times_unit_BC = (
            BC
            / (((BC[:, 0]) ** 2 + (BC[:, 1]) ** 2) ** 0.5)[:, None]
            * BA_dot_unit_BC[:, None]
        )

        dx_distance[positive_index] = (
            (BA[positive_index, 0] - BA_dot_unit_BC_times_unit_BC[positive_index, 0])
            ** 2
            + (BA[positive_index, 1] - BA_dot_unit_BC_times_unit_BC[positive_index, 1])
            ** 2
        ) ** 0.5
        dx_distance[negative_index] = np.minimum(
            ((CA[negative_index, 0]) ** 2 + (CA[negative_index, 1]) ** 2) ** 0.5,
            ((BA[negative_index, 0]) ** 2 + (BA[negative_index, 1]) ** 2) ** 0.5,
        )

        if np.shape(zero_index)[0] != 0:
            dx_distance[zero_index] = np.minimum(
                ((CA[zero_index, 0]) ** 2 + (CA[zero_index, 1]) ** 2) ** 0.5,
                ((BA[zero_index, 0]) ** 2 + (BA[zero_index, 1]) ** 2) ** 0.5,
            )

        min_distance = np.min(dx_distance)

        min_index = np.argmin(dx_distance)

        if (
            min_index in positive_index
        ):  # if the smallest distance is between AD, D in between BC
            x_coor_min_point_segment = (
                BA_dot_unit_BC_times_unit_BC[min_index, 0] + BxByCxCy[min_index, 0]
            )
            y_coor_min_point_segment = (
                BA_dot_unit_BC_times_unit_BC[min_index, 1] + BxByCxCy[min_index, 1]
            )
        if (min_index in negative_index) or (
            min_index in zero_index
        ):  # if the smallest distance is AB or AC
            if ((CA[min_index, 0]) ** 2 + (CA[min_index, 1]) ** 2) ** 0.5 < (
                (BA[min_index, 0]) ** 2 + (BA[min_index, 1]) ** 2
            ) ** 0.5:
                x_coor_min_point_segment = BxByCxCy[min_index, 2]
                y_coor_min_point_segment = BxByCxCy[min_index, 3]
            else:
                x_coor_min_point_segment = BxByCxCy[min_index, 0]
                y_coor_min_point_segment = BxByCxCy[min_index, 1]

        d_distance_dx = (x_G[i, 0] - x_coor_min_point_segment) / min_distance
        d_distance_dy = (x_G[i, 1] - y_coor_min_point_segment) / min_distance

        heaviside_scaling_factor = 4.0e-7

        heaviside = np.tanh((min_distance + 1.0e-15) / heaviside_scaling_factor)

        heaviside_P_x = (
            d_distance_dx
            / heaviside_scaling_factor
            * (1.0 / np.cosh((min_distance + 1.0e-15) / heaviside_scaling_factor)) ** 2
        )  # (1-(np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor))**2)
        heaviside_P_y = (
            d_distance_dy
            / heaviside_scaling_factor
            * (1.0 / np.cosh((min_distance + 1.0e-15) / heaviside_scaling_factor)) ** 2
        )  # (1-(np.tanh((min_distance+1.0e-15)/heaviside_scaling_factor))**2)

    for j in range(np.shape(x_nodes)[0]):

        z_ij = (
            ((x_G[i, 0] - x_nodes[j, 0]) ** 2 + (x_G[i, 1] - x_nodes[j, 1]) ** 2) ** 0.5
        ) / a[j]
        z_ij_P_x = (x_G[i, 0] - x_nodes[j, 0]) / (
            a[j] * z_ij * a[j] + 2.220446049250313e-16
        )  # partial z partial x, add the small number to force the term with machine accuracy
        z_ij_P_y = (x_G[i, 1] - x_nodes[j, 1]) / (
            a[j] * z_ij * a[j] + 2.220446049250313e-16
        )  # partial z partial y

        x_I = x_nodes[j]

        H_scaling_factor = 1.0e-6

        H_T = np.array(
            [
                1,
                (x_G[i][0] - x_I[0]) / H_scaling_factor,
                (x_G[i][1] - x_I[1]) / H_scaling_factor,
            ],
            dtype=np.float64,
        )
        H = np.transpose(H_T)

        HT_P_x = (
            np.array([0, 1, 0], dtype=np.float64) / H_scaling_factor
        )  # partial H partial x
        HT_P_y = (
            np.array([0, 0, 1], dtype=np.float64) / H_scaling_factor
        )  # partial H partial y

        H_P_x = np.transpose(HT_P_x)
        H_P_y = np.transpose(HT_P_y)

        if z_ij >= 0 and z_ij < 0.5:

            phi_ij = 2.0 / 3 - 4 * z_ij**2 + 4 * z_ij**3
            phi_P_z = -8.0 * z_ij + 12.0 * z_ij**2  # partial phi partial z
        else:
            if z_ij <= 1 and z_ij >= 0.5:
                phi_ij = 4.0 / 3 - 4 * z_ij + 4 * z_ij**2 - 4.0 / 3 * z_ij**3
                phi_P_z = -4 + 8 * z_ij - 4 * z_ij**2

        if z_ij >= 0 and z_ij <= 1.0:
            # print('yes')
            # phi_nonzerovalue_data.append(phi_ij)

            node_not_on_interface = "True"

            for i_nodes in range(num_interface_segments * 2):
                # print('yyy')
                if (
                    abs(x_nodes[j, 0] - interface_nodes[i_nodes, 0]) < 1e-10
                    and abs(x_nodes[j, 1] - interface_nodes[i_nodes, 1]) < 1e-10
                ):
                    node_not_on_interface = "False"

            if node_not_on_interface == "True":
                if nodes_grain_id[j] == Gauss_grain_id[i]:

                    phi_nonzero_index_row.append(i)
                    phi_nonzero_index_column.append(j)
                    phi_nonzerovalue_data.append(phi_ij * heaviside)

                    phi_P_x_ij = phi_P_z * z_ij_P_x
                    phi_P_y_ij = phi_P_z * z_ij_P_y
                    phi_P_x_nonzerovalue_data.append(
                        phi_P_x_ij * heaviside + phi_ij * heaviside_P_x
                    )  # partial phi partial x
                    phi_P_y_nonzerovalue_data.append(
                        phi_P_y_ij * heaviside + phi_ij * heaviside_P_y
                    )  # partial phi partial y

                    z.append(z_ij)
                    z_P_x.append(z_ij_P_x)
                    z_P_y.append(z_ij_P_y)
                    phipz.append(phi_P_z)
                    # Note: For interface method with 2D (M shape is 3x3),
                    # phi_P_z_nonzerovalue_data is not populated
                    for ii in range(3):
                        for jj in range(3):
                            M[i][ii][jj] = (
                                M[i][ii][jj] + H[ii] * H_T[jj] * phi_ij * heaviside
                            )
                            M_P_x[i][ii][jj] = (
                                M_P_x[i][ii][jj]
                                + H[ii]
                                * H_T[jj]
                                * (phi_P_x_ij * heaviside + phi_ij * heaviside_P_x)
                                + H_P_x[ii] * H_T[jj] * phi_ij * heaviside
                                + H[ii] * HT_P_x[jj] * phi_ij * heaviside
                            )
                            M_P_y[i][ii][jj] = (
                                M_P_y[i][ii][jj]
                                + H[ii]
                                * H_T[jj]
                                * (phi_P_y_ij * heaviside + phi_ij * heaviside_P_y)
                                + H_P_y[ii] * H_T[jj] * phi_ij * heaviside
                                + H[ii] * HT_P_y[jj] * phi_ij * heaviside
                            )

    return (
        np.array(phi_nonzero_index_row),
        np.array(phi_nonzero_index_column),
        np.array(phi_nonzerovalue_data),
        np.array(phi_P_x_nonzerovalue_data),
        np.array(phi_P_y_nonzerovalue_data),
        np.array(phi_P_z_nonzerovalue_data),
        M,
        M_P_x,
        M_P_y,
        M_P_z,
    )


def compute_z_and_H_2d(x_G_i, x_nodes_j, a_j, H_scaling_factor, eps):
    """Compute z values and H matrices for 2D case (M shape is 3x3)."""

    x_I = x_nodes_j

    z_ij = (
        ((x_G_i[0] - x_nodes_j[0]) ** 2 + (x_G_i[1] - x_nodes_j[1]) ** 2) ** 0.5
    ) / a_j

    z_ij_P_x = (x_G_i[0] - x_nodes_j[0]) / (
        a_j * z_ij * a_j + eps
    )  # partial z partial x
    z_ij_P_y = (x_G_i[1] - x_nodes_j[1]) / (
        a_j * z_ij * a_j + eps
    )  # partial z partial y

    H_T = np.array(
        [
            1,
            (x_G_i[0] - x_I[0]) / H_scaling_factor,
            (x_G_i[1] - x_I[1]) / H_scaling_factor,
        ],
        dtype=np.float64,
    )
    HT_P_x = (
        np.array([0, 1, 0], dtype=np.float64) / H_scaling_factor
    )  # partial H partial x
    HT_P_y = (
        np.array([0, 0, 1], dtype=np.float64) / H_scaling_factor
    )  # partial H partial y

    H = np.transpose(H_T)
    H_P_x = np.transpose(HT_P_x)
    H_P_y = np.transpose(HT_P_y)

    return (
        z_ij,
        z_ij_P_x,
        z_ij_P_y,
        None,
        H_T,
        HT_P_x,
        HT_P_y,
        None,
        H,
        H_P_x,
        H_P_y,
        None,
    )


def compute_z_and_H_3d(x_G_i, x_nodes_j, a_j, H_scaling_factor, eps):
    """Compute z values and H matrices for 3D case (M shape is 4x4)."""

    x_I = x_nodes_j

    z_ij = (
        (
            (x_G_i[0] - x_nodes_j[0]) ** 2
            + (x_G_i[1] - x_nodes_j[1]) ** 2
            + (x_G_i[2] - x_nodes_j[2]) ** 2
        )
        ** 0.5
    ) / a_j

    z_ij_P_x = (x_G_i[0] - x_nodes_j[0]) / (
        a_j * z_ij * a_j + eps
    )  # partial z partial x
    z_ij_P_y = (x_G_i[1] - x_nodes_j[1]) / (
        a_j * z_ij * a_j + eps
    )  # partial z partial y
    z_ij_P_z = (x_G_i[2] - x_nodes_j[2]) / (
        a_j * z_ij * a_j + eps
    )  # partial z partial z

    H_T = np.array(
        [
            1,
            (x_G_i[0] - x_I[0]) / H_scaling_factor,
            (x_G_i[1] - x_I[1]) / H_scaling_factor,
            (x_G_i[2] - x_I[2]) / H_scaling_factor,
        ],
        dtype=np.float64,
    )
    HT_P_x = (
        np.array([0, 1, 0, 0], dtype=np.float64) / H_scaling_factor
    )  # partial H partial x
    HT_P_y = (
        np.array([0, 0, 1, 0], dtype=np.float64) / H_scaling_factor
    )  # partial H partial y
    HT_P_z = (
        np.array([0, 0, 0, 1], dtype=np.float64) / H_scaling_factor
    )  # partial H partial z

    H = np.transpose(H_T)
    H_P_x = np.transpose(HT_P_x)
    H_P_y = np.transpose(HT_P_y)
    H_P_z = np.transpose(HT_P_z)

    return (
        z_ij,
        z_ij_P_x,
        z_ij_P_y,
        z_ij_P_z,
        H_T,
        HT_P_x,
        HT_P_y,
        HT_P_z,
        H,
        H_P_x,
        H_P_y,
        H_P_z,
    )


def compute_phi_M_standard(
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
):
    """Compute phi_M using standard method (else case)."""

    if M_P_z is None:
        # Initialize with zeros or correct shape
        M_P_z = np.zeros_like(M)  # Adjust shape accordingly

    phi_nonzero_index_row = []
    phi_nonzero_index_column = []
    phi_nonzerovalue_data = []
    phi_P_x_nonzerovalue_data = []
    phi_P_y_nonzerovalue_data = []
    phi_P_z_nonzerovalue_data = []
    z = []
    z_P_x = []
    z_P_y = []
    z_P_z = []
    phipz = []

    H_scaling_factor = 1.0e-6
    eps = 2.220446049250313e-16

    for i in range(np.shape(x_G)[0]):
        for j in range(np.shape(x_nodes)[0]):

            # Call appropriate function based on dimension
            if np.shape(M)[1] == 3:
                # 2D case
                (
                    z_ij,
                    z_ij_P_x,
                    z_ij_P_y,
                    z_ij_P_z,
                    H_T,
                    HT_P_x,
                    HT_P_y,
                    HT_P_z,
                    H,
                    H_P_x,
                    H_P_y,
                    H_P_z,
                ) = compute_z_and_H_2d(x_G[i], x_nodes[j], a[j], H_scaling_factor, eps)
            elif np.shape(M)[1] == 4:
                # 3D case
                (
                    z_ij,
                    z_ij_P_x,
                    z_ij_P_y,
                    z_ij_P_z,
                    H_T,
                    HT_P_x,
                    HT_P_y,
                    HT_P_z,
                    H,
                    H_P_x,
                    H_P_y,
                    H_P_z,
                ) = compute_z_and_H_3d(x_G[i], x_nodes[j], a[j], H_scaling_factor, eps)
            else:
                raise ValueError(f"Unsupported M shape: {np.shape(M)[1]}")

            if z_ij >= 0 and z_ij < 0.5:

                phi_ij = 2.0 / 3 - 4 * z_ij**2 + 4 * z_ij**3
                phi_P_z = -8.0 * z_ij + 12.0 * z_ij**2  # partial phi partial z
            else:
                if z_ij <= 1 and z_ij >= 0.5:
                    phi_ij = 4.0 / 3 - 4 * z_ij + 4 * z_ij**2 - 4.0 / 3 * z_ij**3
                    phi_P_z = -4 + 8 * z_ij - 4 * z_ij**2

            if z_ij >= 0 and z_ij <= 1.0:

                phi_nonzerovalue_data.append(phi_ij)
                phi_nonzero_index_row.append(i)
                phi_nonzero_index_column.append(j)
                phi_P_x_ij = phi_P_z * z_ij_P_x
                phi_P_y_ij = phi_P_z * z_ij_P_y
                if np.shape(M)[1] == 4:
                    phi_P_z_ij = phi_P_z * z_ij_P_z
                    phi_P_z_nonzerovalue_data.append(phi_P_z_ij)
                    z_P_z.append(z_ij_P_z)

                phi_P_x_nonzerovalue_data.append(phi_P_x_ij)  # partial phi partial x
                phi_P_y_nonzerovalue_data.append(phi_P_y_ij)  # partial phi partial y
                z.append(z_ij)
                z_P_x.append(z_ij_P_x)
                z_P_y.append(z_ij_P_y)
                phipz.append(phi_P_z)

                for ii in range(np.shape(M)[1]):
                    for jj in range(np.shape(M)[1]):
                        # if i==13:
                        #     print(M[i])
                        M[i][ii][jj] = M[i][ii][jj] + H[ii] * H_T[jj] * phi_ij
                        M_P_x[i][ii][jj] = (
                            M_P_x[i][ii][jj]
                            + H[ii] * H_T[jj] * phi_P_x_ij
                            + H_P_x[ii] * H_T[jj] * phi_ij
                            + H[ii] * HT_P_x[jj] * phi_ij
                        )
                        M_P_y[i][ii][jj] = (
                            M_P_y[i][ii][jj]
                            + H[ii] * H_T[jj] * phi_P_y_ij
                            + H_P_y[ii] * H_T[jj] * phi_ij
                            + H[ii] * HT_P_y[jj] * phi_ij
                        )
                        if np.shape(M)[1] == 4:
                            M_P_z[i][ii][jj] = (
                                M_P_z[i][ii][jj]
                                + H[ii] * H_T[jj] * phi_P_z_ij
                                + H_P_z[ii] * H_T[jj] * phi_ij
                                + H[ii] * HT_P_z[jj] * phi_ij
                            )

    return (
        np.array(phi_nonzero_index_row),
        np.array(phi_nonzero_index_column),
        np.array(phi_nonzerovalue_data),
        np.array(phi_P_x_nonzerovalue_data),
        np.array(phi_P_y_nonzerovalue_data),
        np.array(phi_P_z_nonzerovalue_data),
        M,
        M_P_x,
        M_P_y,
        M_P_z,
    )


def compute_phi_M(
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
    M_P_z=None,
    use_vectorized=True,
    dtype=np.float64,
):
    """
    Compute RKPM kernel values φ and moment matrices M at all Gauss points.

    This function evaluates the cubic B-spline kernel for all Gauss point-node
    pairs within the support radius (z ≤ 1) and accumulates the moment matrices
    needed for shape function computation.

    The moment matrix at Gauss point x is:

        M(x) = Σ_I H(x - x_I) H^T(x - x_I) φ(z_I)

    where the sum is over all nodes I with z_I ≤ 1.

    Parameters
    ----------
    x_G : np.ndarray
        Gauss point coordinates, shape (n_gauss, dim).
    Gauss_grain_id : np.ndarray
        Phase ID at each Gauss point, shape (n_gauss,).
    x_nodes : np.ndarray
        Node coordinates, shape (n_nodes, dim).
    nodes_grain_id : np.ndarray
        Phase ID at each node, shape (n_nodes,).
    a : np.ndarray
        Support radius for each node, shape (n_nodes,).
    M : np.ndarray
        Moment matrix array to accumulate, shape (n_gauss, basis, basis).
        For 3D: basis=4 (linear), For 2D: basis=3.
    M_P_x, M_P_y : np.ndarray
        Partial derivatives of M w.r.t. x, y. Same shape as M.
    num_interface_segments : int
        Number of interface segments (for IM-RKPM method).
    interface_nodes : np.ndarray
        Coordinates of nodes on interfaces.
    BxByCxCy : np.ndarray
        Interface segment endpoint coordinates.
    IM_RKPM : str
        "True" for interface-modified RKPM, "False" otherwise.
    single_grain : str
        "True" for single grain, "False" for multi-phase.
    M_P_z : np.ndarray, optional
        Partial derivative of M w.r.t. z (3D only).
    use_vectorized : bool, optional
        If True, use optimized vectorized implementation. Default True.
    dtype : numpy dtype, optional
        Data type for computations. Default np.float64.

    Returns
    -------
    phi_nonzero_index_row : np.ndarray
        Gauss point indices for non-zero kernel values, shape (n_nnz,).
    phi_nonzero_index_column : np.ndarray
        Node indices for non-zero kernel values, shape (n_nnz,).
    phi_nonzerovalue_data : np.ndarray
        Kernel values φ(z) at non-zero pairs, shape (n_nnz,).
    phi_P_x_nonzerovalue_data : np.ndarray
        ∂φ/∂x at non-zero pairs, shape (n_nnz,).
    phi_P_y_nonzerovalue_data : np.ndarray
        ∂φ/∂y at non-zero pairs, shape (n_nnz,).
    phi_P_z_nonzerovalue_data : np.ndarray
        ∂φ/∂z at non-zero pairs (3D), shape (n_nnz,).
    M : np.ndarray
        Updated moment matrix, shape (n_gauss, basis, basis).
    M_P_x, M_P_y, M_P_z : np.ndarray
        Updated moment matrix derivatives.
    """

    if single_grain == "False" and IM_RKPM == "True":
        # Use interface method
        return compute_phi_M_with_interface_method(
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
            M_P_z,
        )
    else:
        # Use standard method - with sparse optimization if available
        if use_vectorized and SPARSE_PHI_AVAILABLE:
            print(f"using SPARSE phi_m_standard with dtype={dtype}", flush=True)
            return compute_phi_M_standard_sparse(
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
                M_P_z,
                dtype,
            )
        else:
            return compute_phi_M_standard(
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
                M_P_z,
            )


def shape_grad_shape_func(
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
    use_vectorized=True,
    dtype=np.float64,
):
    """
    Compute RKPM shape functions and their gradients.

    Evaluates the shape function Ψ_I(x) and its spatial derivatives for all
    non-zero Gauss-node pairs. The shape function is:

        Ψ_I(x) = H^T(0) M^{-1}(x) H(x - x_I) φ(z)

    Gradients are computed using the direct differentiation method:

        ∂Ψ/∂x = H^T M^{-1} H ∂φ/∂x + H^T ∂M^{-1}/∂x H φ + H^T M^{-1} ∂H/∂x φ

    Parameters
    ----------
    x_G : np.ndarray
        Gauss point coordinates, shape (n_gauss, dim).
    x_nodes : np.ndarray
        Node coordinates, shape (n_nodes, dim).
    num_non_zero_phi_a : int
        Number of non-zero kernel values.
    HT0 : np.ndarray
        Polynomial basis at origin [1, 0, 0, 0], shape (basis,).
    M : np.ndarray
        Moment matrices at Gauss points, shape (n_gauss, basis, basis).
    M_P_x, M_P_y : np.ndarray
        Moment matrix derivatives, same shape as M.
    differential_method : str
        "direct" for direct differentiation, "implicite" for implicit.
    HT1, HT2 : np.ndarray
        Basis vectors for implicit method derivatives.
    phi_nonzerovalue_data : np.ndarray
        Non-zero kernel values, shape (n_nnz,).
    phi_P_*_nonzerovalue_data : np.ndarray
        Kernel derivatives, shape (n_nnz,).
    phi_nonzero_index_row, phi_nonzero_index_column : np.ndarray
        Sparse indices for non-zero pairs, shape (n_nnz,).
    det_J_time_weight : np.ndarray
        Jacobian × Gauss weight at each Gauss point, shape (n_gauss,).
    IM_RKPM : str
        "True" for interface-modified RKPM.
    M_P_z : np.ndarray, optional
        Moment matrix z-derivative (3D).
    HT3 : np.ndarray, optional
        Basis vector for implicit z-derivative.
    phi_P_z_nonzerovalue_data : np.ndarray, optional
        Kernel z-derivative values.
    use_vectorized : bool, optional
        Use vectorized implementation. Default True.
    dtype : numpy dtype, optional
        Data type for computations. Default np.float64.

    Returns
    -------
    shape_func_value : np.ndarray
        Shape function values, shape (n_nnz,).
    shape_func_times_det_J_time_weight_value : np.ndarray
        Weighted shape functions, shape (n_nnz,).
    grad_shape_func_x_value : np.ndarray
        ∂Ψ/∂x values, shape (n_nnz,).
    grad_shape_func_y_value : np.ndarray
        ∂Ψ/∂y values, shape (n_nnz,).
    grad_shape_func_z_value : np.ndarray
        ∂Ψ/∂z values (3D), shape (n_nnz,).
    grad_shape_func_*_times_det_J_time_weight_value : np.ndarray
        Weighted gradient values, shape (n_nnz,).
    """

    # Use vectorized version if available and requested
    if use_vectorized and VECTORIZED_GRAD_AVAILABLE:
        print(f"using vectorized shape_grad_shape_func with dtype={dtype}", flush=True)
        return shape_grad_shape_func_vectorized(
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
            M_P_z,
            HT3,
            phi_P_z_nonzerovalue_data,
            dtype,
        )

    # Otherwise use original implementation
    shape_func_value = []
    shape_func_times_det_J_time_weight_value = []
    grad_shape_func_x_value = []
    grad_shape_func_y_value = []
    grad_shape_func_z_value = []
    grad_shape_func_x_times_det_J_time_weight_value = []
    grad_shape_func_y_times_det_J_time_weight_value = []
    grad_shape_func_z_times_det_J_time_weight_value = []

    for ii in range(num_non_zero_phi_a):

        i = int(phi_nonzero_index_row[ii])
        j = int(phi_nonzero_index_column[ii])

        # compute the shape function and the gradient of shape function
        x_I = x_nodes[j]

        H_scaling_factor = 1.0e-6
        if np.shape(M)[1] == 3:
            H_T = np.array(
                [
                    1,
                    (x_G[i][0] - x_I[0]) / H_scaling_factor,
                    (x_G[i][1] - x_I[1]) / H_scaling_factor,
                ],
                dtype=np.float64,
            )
            HT_P_x = (
                np.array([0, 1, 0], dtype=np.float64) / H_scaling_factor
            )  # partial H partial x
            HT_P_y = (
                np.array([0, 0, 1], dtype=np.float64) / H_scaling_factor
            )  # partial H partial y

        if np.shape(M)[1] == 4:

            H_T = np.array(
                [
                    1,
                    (x_G[i][0] - x_I[0]) / H_scaling_factor,
                    (x_G[i][1] - x_I[1]) / H_scaling_factor,
                    (x_G[i][2] - x_I[2]) / H_scaling_factor,
                ],
                dtype=np.float64,
            )
            HT_P_x = (
                np.array([0, 1, 0, 0], dtype=np.float64) / H_scaling_factor
            )  # partial H partial x
            HT_P_y = (
                np.array([0, 0, 1, 0], dtype=np.float64) / H_scaling_factor
            )  # partial H partial y
            HT_P_z = (
                np.array([0, 0, 0, 1], dtype=np.float64) / H_scaling_factor
            )  # partial H partial y
            H_P_z = np.transpose(HT_P_z)

        H = np.transpose(H_T)

        H_P_x = np.transpose(HT_P_x)
        H_P_y = np.transpose(HT_P_y)

        # print(i, x_G[i,:]) #13 [0.00000000e+00 2.02113249e-05 1.07886751e-05]

        shape_func_ij = (
            np.dot(
                (
                    np.dot(
                        (HT0).astype(np.float64),
                        (np.linalg.inv(M[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H.astype(np.float64),
            )
            * phi_nonzerovalue_data[ii]
        )

        if differential_method == "implicite" and IM_RKPM == "False":
            grad_shape_func_x_ij = (
                np.dot(
                    (
                        np.dot(
                            (HT1).astype(np.float64),
                            (np.linalg.inv(M[i])).astype(np.float64),
                        )
                    ).astype(np.float64),
                    H.astype(np.float64),
                )
                * phi_nonzerovalue_data[ii]
            )
            grad_shape_func_y_ij = (
                np.dot(
                    (
                        np.dot(
                            (HT2).astype(np.float64),
                            (np.linalg.inv(M[i])).astype(np.float64),
                        )
                    ).astype(np.float64),
                    H.astype(np.float64),
                )
                * phi_nonzerovalue_data[ii]
            )
            if np.shape(M) == 4:
                grad_shape_func_z_ij = (
                    np.dot(
                        (
                            np.dot(
                                (HT3).astype(np.float64),
                                (np.linalg.inv(M[i])).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H.astype(np.float64),
                    )
                    * phi_nonzerovalue_data[ii]
                )

        else:
            if differential_method == "direct" or IM_RKPM == "True":
                M_inv_P_x_i = -np.dot(
                    np.dot(
                        np.linalg.inv(M[i].astype(np.float64)).astype(np.float64),
                        M_P_x[i].astype(np.float64),
                    ),
                    np.linalg.inv(M[i].astype(np.float64)).astype(np.float64),
                )
                M_inv_P_y_i = -np.dot(
                    np.dot(
                        np.linalg.inv(M[i].astype(np.float64)).astype(np.float64),
                        M_P_y[i].astype(np.float64),
                    ),
                    np.linalg.inv(M[i].astype(np.float64)).astype(np.float64),
                )
                grad_shape_func_x_ij = (
                    np.dot(
                        (
                            np.dot(
                                (HT0).astype(np.float64),
                                (np.linalg.inv(M[i])).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H.astype(np.float64),
                    )
                    * phi_P_x_nonzerovalue_data[ii]
                    + np.dot(
                        (
                            np.dot(
                                (HT0).astype(np.float64),
                                (M_inv_P_x_i).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H.astype(np.float64),
                    )
                    * phi_nonzerovalue_data[ii]
                    + np.dot(
                        (
                            np.dot(
                                (HT0).astype(np.float64),
                                (np.linalg.inv(M[i])).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H_P_x.astype(np.float64),
                    )
                    * phi_nonzerovalue_data[ii]
                )
                grad_shape_func_y_ij = (
                    np.dot(
                        (
                            np.dot(
                                (HT0).astype(np.float64),
                                (np.linalg.inv(M[i])).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H.astype(np.float64),
                    )
                    * phi_P_y_nonzerovalue_data[ii]
                    + np.dot(
                        (
                            np.dot(
                                (HT0).astype(np.float64),
                                (M_inv_P_y_i).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H.astype(np.float64),
                    )
                    * phi_nonzerovalue_data[ii]
                    + np.dot(
                        (
                            np.dot(
                                (HT0).astype(np.float64),
                                (np.linalg.inv(M[i])).astype(np.float64),
                            )
                        ).astype(np.float64),
                        H_P_y.astype(np.float64),
                    )
                    * phi_nonzerovalue_data[ii]
                )
                if np.shape(M)[1] == 4:
                    M_inv_P_z_i = -np.dot(
                        np.dot(
                            np.linalg.inv(M[i].astype(np.float64)).astype(np.float64),
                            M_P_z[i].astype(np.float64),
                        ),
                        np.linalg.inv(M[i].astype(np.float64)).astype(np.float64),
                    )
                    grad_shape_func_z_ij = (
                        np.dot(
                            (
                                np.dot(
                                    (HT0).astype(np.float64),
                                    (np.linalg.inv(M[i])).astype(np.float64),
                                )
                            ).astype(np.float64),
                            H.astype(np.float64),
                        )
                        * phi_P_z_nonzerovalue_data[ii]
                        + np.dot(
                            (
                                np.dot(
                                    (HT0).astype(np.float64),
                                    (M_inv_P_z_i).astype(np.float64),
                                )
                            ).astype(np.float64),
                            H.astype(np.float64),
                        )
                        * phi_nonzerovalue_data[ii]
                        + np.dot(
                            (
                                np.dot(
                                    (HT0).astype(np.float64),
                                    (np.linalg.inv(M[i])).astype(np.float64),
                                )
                            ).astype(np.float64),
                            H_P_z.astype(np.float64),
                        )
                        * phi_nonzerovalue_data[ii]
                    )

            else:
                print("differential method is not defined", flush=True)
        shape_func_value.append(shape_func_ij)
        grad_shape_func_x_value.append(grad_shape_func_x_ij)
        grad_shape_func_y_value.append(grad_shape_func_y_ij)

        shape_func_times_det_J_time_weight_value.append(
            shape_func_ij * det_J_time_weight[i]
        )
        grad_shape_func_x_times_det_J_time_weight_value.append(
            grad_shape_func_x_ij * det_J_time_weight[i]
        )
        grad_shape_func_y_times_det_J_time_weight_value.append(
            grad_shape_func_y_ij * det_J_time_weight[i]
        )

        if np.shape(M)[1] == 4:
            grad_shape_func_z_value.append(grad_shape_func_z_ij)
            grad_shape_func_z_times_det_J_time_weight_value.append(
                grad_shape_func_z_ij * det_J_time_weight[i]
            )

    return (
        np.array(shape_func_value),
        np.array(shape_func_times_det_J_time_weight_value),
        np.array(grad_shape_func_x_value),
        np.array(grad_shape_func_y_value),
        np.array(grad_shape_func_z_value),
        np.array(grad_shape_func_x_times_det_J_time_weight_value),
        np.array(grad_shape_func_y_times_det_J_time_weight_value),
        np.array(grad_shape_func_z_times_det_J_time_weight_value),
    )


def shape_func_n_nodes_by_n_nodes(
    x_G,
    x_nodes,
    num_non_zero_phi_a,
    HT0,
    M,
    phi_nonzerovalue_data,
    phi_nonzero_index_row,
    phi_nonzero_index_column,
    use_vectorized=True,
):
    """Compute shape functions for non-zero phi values.

    Args:
        x_G: Gauss points
        x_nodes: Node points
        num_non_zero_phi_a: Number of non-zero phi values
        HT0: Initial H transpose matrix
        M: Moment matrices
        phi_nonzerovalue_data: Non-zero phi values
        phi_nonzero_index_row: Row indices for non-zero phi
        phi_nonzero_index_column: Column indices for non-zero phi
        use_vectorized: Whether to use vectorized implementation if available

    Returns:
        List of shape function values
    """
    # Use vectorized version if available and requested
    if use_vectorized:
        print(f"Using vectorized shape function", flush=True)
        return shape_func_n_nodes_by_n_nodes_vectorized(
            x_G,
            x_nodes,
            num_non_zero_phi_a,
            HT0,
            M,
            phi_nonzerovalue_data,
            phi_nonzero_index_row,
            phi_nonzero_index_column,
        )

    # Fall back to original implementation
    shape_func_value = []

    for ii in range(num_non_zero_phi_a):
        i = int(phi_nonzero_index_row[ii])
        j = int(phi_nonzero_index_column[ii])

        # compute the shape function and the gradient of shape function
        x_I = x_nodes[j]

        H_scaling_factor = 1.0e-6
        if np.shape(M)[1] == 3:
            H_T = np.array(
                [
                    1,
                    (x_G[i][0] - x_I[0]) / H_scaling_factor,
                    (x_G[i][1] - x_I[1]) / H_scaling_factor,
                ],
                dtype=np.float64,
            )
        if np.shape(M)[1] == 4:
            H_T = np.array(
                [
                    1,
                    (x_G[i][0] - x_I[0]) / H_scaling_factor,
                    (x_G[i][1] - x_I[1]) / H_scaling_factor,
                    (x_G[i][2] - x_I[2]) / H_scaling_factor,
                ],
                dtype=np.float64,
            )

        H = np.transpose(H_T)

        shape_func_ij = (
            np.dot(
                (
                    np.dot(
                        (HT0).astype(np.float64),
                        (np.linalg.inv(M[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H.astype(np.float64),
            )
            * phi_nonzerovalue_data[ii]
        )

        shape_func_value.append(shape_func_ij)

    return shape_func_value


def shape_func_n_nodes_by_n_nodes_vectorized(
    x_G,
    x_nodes,
    num_non_zero_phi_a,
    HT0,
    M,
    phi_nonzerovalue_data,
    phi_nonzero_index_row,
    phi_nonzero_index_column,
):
    """Vectorized computation of shape functions for non-zero phi values.

    Args:
        x_G: Gauss points array (n_gauss, dim)
        x_nodes: Node points array (n_nodes, dim)
        num_non_zero_phi_a: Number of non-zero phi values
        HT0: Initial H transpose matrix
        M: Moment matrices (n_gauss, basis_size, basis_size)
        phi_nonzerovalue_data: Non-zero phi values
        phi_nonzero_index_row: Row indices for non-zero phi
        phi_nonzero_index_column: Column indices for non-zero phi

    Returns:
        numpy.ndarray: Array of shape function values of shape (num_non_zero_phi_a,)
    """
    # Convert indices to integer arrays
    row_indices = phi_nonzero_index_row[:num_non_zero_phi_a].astype(int)
    col_indices = phi_nonzero_index_column[:num_non_zero_phi_a].astype(int)

    # Get relevant points
    x_G_sparse = x_G[row_indices]  # (num_non_zero, dim)
    x_nodes_sparse = x_nodes[col_indices]  # (num_non_zero, dim)

    # Compute differences
    diff = x_G_sparse - x_nodes_sparse  # (num_non_zero, dim)

    # Determine dimension and basis size
    dim = x_G.shape[1] if len(x_G.shape) > 1 else 1
    basis_size = M.shape[1]

    H_scaling_factor = 1.0e-6

    # Build H_T matrices based on dimension
    if basis_size == 3:  # 2D case
        # H_T shape: (num_non_zero, 3)
        H_T = np.zeros((num_non_zero_phi_a, 3), dtype=np.float64)
        H_T[:, 0] = 1.0
        H_T[:, 1] = diff[:, 0] / H_scaling_factor
        H_T[:, 2] = diff[:, 1] / H_scaling_factor
    elif basis_size == 4:  # 3D case
        # H_T shape: (num_non_zero, 4)
        H_T = np.zeros((num_non_zero_phi_a, 4), dtype=np.float64)
        H_T[:, 0] = 1.0
        H_T[:, 1] = diff[:, 0] / H_scaling_factor
        H_T[:, 2] = diff[:, 1] / H_scaling_factor
        H_T[:, 3] = diff[:, 2] / H_scaling_factor
    else:
        raise ValueError(f"Unsupported basis size: {basis_size}")

    # In the original code, H = np.transpose(H_T) where H_T is a 1D array
    # This doesn't change the shape for 1D arrays in numpy
    # We need H to be shape (num_non_zero, basis_size) for vectorized dot product
    H = H_T  # Keep as (num_non_zero, basis_size)

    # Get relevant M matrices and compute inverses
    M_sparse = M[row_indices]  # (num_non_zero, basis_size, basis_size)

    # Compute M inverses for all relevant matrices at once
    M_inv = np.linalg.inv(
        M_sparse.astype(np.float64)
    )  # (num_non_zero, basis_size, basis_size)

    # Compute shape function values using einsum for efficiency
    # HT0 is (basis_size,) or (1, basis_size), M_inv is (num_non_zero, basis_size, basis_size)
    # H is (num_non_zero, basis_size)

    # Ensure HT0 is 2D with shape (1, basis_size)
    if HT0.ndim == 1:
        HT0_2d = HT0.reshape(1, -1)  # Convert from (basis_size,) to (1, basis_size)
    else:
        HT0_2d = HT0  # Already 2D

    # First compute HT0 @ M_inv for each matrix
    # We need to broadcast HT0_2d to work with multiple M_inv matrices
    # HT0_2d is (1, basis_size), M_inv is (num_non_zero, basis_size, basis_size)
    # We want result shape: (num_non_zero, 1, basis_size)
    HT0_M_inv = np.matmul(HT0_2d, M_inv)

    # Then compute (HT0 @ M_inv) @ H for each pair
    # We need to do element-wise dot product of HT0_M_inv and H
    # HT0_M_inv is (num_non_zero, 1, basis_size), H is (num_non_zero, basis_size)
    # Use einsum for clarity: 'nib,nb->n'
    shape_func_base = np.einsum("nib,nb->n", HT0_M_inv, H)

    # Get relevant phi values
    phi_values = phi_nonzerovalue_data[:num_non_zero_phi_a]

    # Compute final shape function values
    shape_func_value = shape_func_base * phi_values

    # Return the numpy array directly - no conversion needed
    return shape_func_value
