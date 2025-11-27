import time

start_time = time.time()
import numpy as np
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csr_array
from scipy.sparse.linalg import eigs, spsolve

# Try to import the vectorized version if available
try:
    from shape_function_vectorized import compute_phi_M_standard_vectorized

    VECTORIZED_AVAILABLE = True
except ImportError:
    VECTORIZED_AVAILABLE = False


# @jit  # Disabled due to Numba type inference issues with empty lists
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

    print(f"Compute phi_M using interface method")

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

    print("get in")
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


# @jit  # Disabled due to Numba type inference issues
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


# @jit  # Disabled due to Numba type inference issues
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


# @jit  # Disabled due to Numba type inference issues
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


# @jit  # Not needed for wrapper function that just delegates
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
    use_vectorized=True,  # New parameter to control vectorization
):
    """Main compute_phi_M function that delegates to specialized functions based on conditions.

    Args:
        use_vectorized: If True and vectorized version is available, use the vectorized impl
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
        # Use standard method - with optional vectorization
        if use_vectorized and VECTORIZED_AVAILABLE:
            print(f"using vectorized phi_m_standard")
            return compute_phi_M_standard_vectorized(
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


# @jit  # this is taking so long time, we are vectorizing this part
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
):
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
                print("differential method is not defined")
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
        shape_func_value,
        shape_func_times_det_J_time_weight_value,
        grad_shape_func_x_value,
        grad_shape_func_y_value,
        grad_shape_func_z_value,
        grad_shape_func_x_times_det_J_time_weight_value,
        grad_shape_func_y_times_det_J_time_weight_value,
        grad_shape_func_z_times_det_J_time_weight_value,
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
):
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
