import time

start_time = time.time()
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm


@jit
def func_1(a1, a3, a2, a4, a, M1, M2, M3, a6_num, a5, a6):

    n1 = []
    n2 = []
    n3 = []
    n4 = []
    n5 = []

    z = []
    z_P_x = []
    z_P_y = []
    phipz = []

    for i in range(np.shape(a1)[0]):

        dx_distance = np.zeros(a6_num)

        BA = a1[i, :] - a6[:, :2]
        BC = a6[:, 2:4] - a6[:, :2]
        CB = -a6[:, 2:4] + a6[:, :2]
        CA = a1[i, :] - a6[:, 2:4]

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
                BA_dot_unit_BC_times_unit_BC[min_index, 0] + a6[min_index, 0]
            )
            y_coor_min_point_segment = (
                BA_dot_unit_BC_times_unit_BC[min_index, 1] + a6[min_index, 1]
            )
        if (min_index in negative_index) or (
            min_index in zero_index
        ):  # if the smallest distance is AB or AC
            if ((CA[min_index, 0]) ** 2 + (CA[min_index, 1]) ** 2) ** 0.5 < (
                (BA[min_index, 0]) ** 2 + (BA[min_index, 1]) ** 2
            ) ** 0.5:
                x_coor_min_point_segment = a6[min_index, 2]
                y_coor_min_point_segment = a6[min_index, 3]
            else:
                x_coor_min_point_segment = a6[min_index, 0]
                y_coor_min_point_segment = a6[min_index, 1]

        d_distance_dx = (a1[i, 0] - x_coor_min_point_segment) / min_distance
        d_distance_dy = (a1[i, 1] - y_coor_min_point_segment) / min_distance

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

        for j in range(np.shape(a2)[0]):

            z_ij = (
                ((a1[i, 0] - a2[j, 0]) ** 2 + (a1[i, 1] - a2[j, 1]) ** 2) ** 0.5
            ) / a[j]
            z_ij_P_x = (a1[i, 0] - a2[j, 0]) / (
                a[j] * z_ij * a[j] + 2.220446049250313e-16
            )  # partial z partial x, add the small number to force the term with machine accuracy
            z_ij_P_y = (a1[i, 1] - a2[j, 1]) / (
                a[j] * z_ij * a[j] + 2.220446049250313e-16
            )  # partial z partial y

            x_I = a2[j]

            H_sacling_factor = 1.0e-6

            H_T = np.array(
                [
                    1,
                    (a1[i][0] - x_I[0]) / H_sacling_factor,
                    (a1[i][1] - x_I[1]) / H_sacling_factor,
                ],
                dtype=np.float64,
            )
            H = np.transpose(H_T)

            HT_P_x = (
                np.array([0, 1, 0], dtype=np.float64) / H_sacling_factor
            )  # partial H partial x
            HT_P_y = (
                np.array([0, 0, 1], dtype=np.float64) / H_sacling_factor
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
                # n3.append(phi_ij)

                node_not_on_interface = "True"

                for i_nodes in range(a6_num * 2):
                    # print('yyy')
                    if (
                        abs(a2[j, 0] - a5[i_nodes, 0]) < 1e-10
                        and abs(a2[j, 1] - a5[i_nodes, 1]) < 1e-10
                    ):
                        node_not_on_interface = "False"

                if node_not_on_interface == "True":
                    if a4[j] == a3[i]:

                        n1.append(i)
                        n2.append(j)
                        n3.append(phi_ij * heaviside)

                        phi_P_x_ij = phi_P_z * z_ij_P_x
                        phi_P_y_ij = phi_P_z * z_ij_P_y
                        n4.append(
                            phi_P_x_ij * heaviside + phi_ij * heaviside_P_x
                        )  # partial phi partial x
                        n5.append(
                            phi_P_y_ij * heaviside + phi_ij * heaviside_P_y
                        )  # partial phi partial y

                        z.append(z_ij)
                        z_P_x.append(z_ij_P_x)
                        z_P_y.append(z_ij_P_y)
                        phipz.append(phi_P_z)
                        for ii in range(3):
                            for jj in range(3):
                                M1[i][ii][jj] = (
                                    M1[i][ii][jj] + H[ii] * H_T[jj] * phi_ij * heaviside
                                )
                                M2[i][ii][jj] = (
                                    M2[i][ii][jj]
                                    + H[ii]
                                    * H_T[jj]
                                    * (phi_P_x_ij * heaviside + phi_ij * heaviside_P_x)
                                    + H_P_x[ii] * H_T[jj] * phi_ij * heaviside
                                    + H[ii] * HT_P_x[jj] * phi_ij * heaviside
                                )
                                M3[i][ii][jj] = (
                                    M3[i][ii][jj]
                                    + H[ii]
                                    * H_T[jj]
                                    * (phi_P_y_ij * heaviside + phi_ij * heaviside_P_y)
                                    + H_P_y[ii] * H_T[jj] * phi_ij * heaviside
                                    + H[ii] * HT_P_y[jj] * phi_ij * heaviside
                                )

                else:
                    n3.append(phi_ij)
                    n1.append(i)
                    n2.append(j)
                    phi_P_x_ij = phi_P_z * z_ij_P_x
                    phi_P_y_ij = phi_P_z * z_ij_P_y
                    n4.append(phi_P_x_ij)  # partial phi partial x
                    n5.append(phi_P_y_ij)  # partial phi partial y
                    z.append(z_ij)
                    z_P_x.append(z_ij_P_x)
                    z_P_y.append(z_ij_P_y)
                    phipz.append(phi_P_z)
                    for ii in range(3):
                        for jj in range(3):
                            M1[i][ii][jj] = M1[i][ii][jj] + H[ii] * H_T[jj] * phi_ij
                            M2[i][ii][jj] = (
                                M2[i][ii][jj]
                                + H[ii] * H_T[jj] * phi_P_x_ij
                                + H_P_x[ii] * H_T[jj] * phi_ij
                                + H[ii] * HT_P_x[jj] * phi_ij
                            )
                            M3[i][ii][jj] = (
                                M3[i][ii][jj]
                                + H[ii] * H_T[jj] * phi_P_y_ij
                                + H_P_y[ii] * H_T[jj] * phi_ij
                                + H[ii] * HT_P_y[jj] * phi_ij
                            )

    return n1, n2, n3, n4, n5, M1, M2, M3


def func_2(a1, a2, n1_num, H0, M1, M2, M3, n3, n4, n5, n1, n2):
    s1 = []
    s2 = []
    s3 = []

    for ii in range(n1_num):
        i = n1[ii]
        j = n2[ii]

        # compute the shape function and the gradient of shape function
        x_I = a2[j]

        H_sacling_factor = 1.0e-6

        H_T = np.array(
            [
                1,
                (a1[i][0] - x_I[0]) / H_sacling_factor,
                (a1[i][1] - x_I[1]) / H_sacling_factor,
            ],
            dtype=np.float64,
        )
        H = np.transpose(H_T)

        HT_P_x = (
            np.array([0, 1, 0], dtype=np.float64) / H_sacling_factor
        )  # partial H partial x
        HT_P_y = (
            np.array([0, 0, 1], dtype=np.float64) / H_sacling_factor
        )  # partial H partial y

        H_P_x = np.transpose(HT_P_x)
        H_P_y = np.transpose(HT_P_y)

        s1_ij = (
            np.dot(
                (
                    np.dot(
                        (H0).astype(np.float64),
                        (np.linalg.inv(M1[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H.astype(np.float64),
            )
            * n3[ii]
        )

        M_inv_2 = -np.dot(
            np.dot(
                np.linalg.inv(M1[i].astype(np.float64)).astype(np.float64),
                M2[i].astype(np.float64),
            ),
            np.linalg.inv(M1[i].astype(np.float64)).astype(np.float64),
        )
        M_inv_3 = -np.dot(
            np.dot(
                np.linalg.inv(M1[i].astype(np.float64)).astype(np.float64),
                M3[i].astype(np.float64),
            ),
            np.linalg.inv(M1[i].astype(np.float64)).astype(np.float64),
        )
        s2_ij = (
            np.dot(
                (
                    np.dot(
                        (H0).astype(np.float64),
                        (np.linalg.inv(M1[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H.astype(np.float64),
            )
            * n4[ii]
            + np.dot(
                (np.dot((H0).astype(np.float64), (M_inv_2).astype(np.float64))).astype(
                    np.float64
                ),
                H.astype(np.float64),
            )
            * n3[ii]
            + np.dot(
                (
                    np.dot(
                        (H0).astype(np.float64),
                        (np.linalg.inv(M1[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H_P_x.astype(np.float64),
            )
            * n3[ii]
        )
        s3_ij = (
            np.dot(
                (
                    np.dot(
                        (H0).astype(np.float64),
                        (np.linalg.inv(M1[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H.astype(np.float64),
            )
            * n5[ii]
            + np.dot(
                (np.dot((H0).astype(np.float64), (M_inv_3).astype(np.float64))).astype(
                    np.float64
                ),
                H.astype(np.float64),
            )
            * n3[ii]
            + np.dot(
                (
                    np.dot(
                        (H0).astype(np.float64),
                        (np.linalg.inv(M1[i])).astype(np.float64),
                    )
                ).astype(np.float64),
                H_P_y.astype(np.float64),
            )
            * n3[ii]
        )

        s1.append(s1_ij)
        s2.append(s2_ij)
        s3.append(s3_ij)

    return s1, s2, s3
