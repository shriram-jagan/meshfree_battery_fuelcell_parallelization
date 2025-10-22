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
):

    phi_nonzero_index_row = []
    phi_nonzero_index_column = []
    phi_nonzerovalue_data = []
    phi_P_x_nonzerovalue_data = []
    phi_P_y_nonzerovalue_data = []
    z = []
    z_P_x = []
    z_P_y = []
    phipz = []
    heaviside_list = []
    heaviside_plotting_points = []
    for i in range(np.shape(x_G)[0]):
        for j in range(np.shape(x_nodes)[0]):

            z_ij = (
                ((x_G[i, 0] - x_nodes[j, 0]) ** 2 + (x_G[i, 1] - x_nodes[j, 1]) ** 2)
                ** 0.5
            ) / a[j]
            z_ij_P_x = (x_G[i, 0] - x_nodes[j, 0]) / (
                a[j] * z_ij * a[j] + 2.220446049250313e-16
            )  # partial z partial x, add the small number to force the term with machine accuracy
            z_ij_P_y = (x_G[i, 1] - x_nodes[j, 1]) / (
                a[j] * z_ij * a[j] + 2.220446049250313e-16
            )  # partial z partial y

            x_I = x_nodes[j]

            H_T = np.array(
                [1, (x_G[i][0] - x_I[0]), (x_G[i][1] - x_I[1])], dtype=np.float64
            )  # scaling factor 1e-6
            H = np.transpose(H_T)

            HT_P_x = np.array([0, 1, 0], dtype=np.float64)  # partial H partial x
            HT_P_y = np.array([0, 0, 1], dtype=np.float64)  # partial H partial y

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
                # if j==855:
                #     print(node_not_on_interface)
                if IM_RKPM == "True" and node_not_on_interface == "True":
                    if Gauss_grain_id[i] == nodes_grain_id[j]:
                        # if j == 855:
                        #     print('error!!!!')
                        # modify the kernal function related to nodes within domain
                        dx_distance = np.zeros(num_interface_segments)
                        # print(np.shape(dx_distance))
                        # print(num_interface_segments)
                        # exit()
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

                        # BA_dot_unit_BC = BA_dot_BC/(((BC[negative_index, 0])**2+(BC[negative_index, 1])**2)**0.5)

                        # BA_dot_unit_BC_times_unit_BC = BC/(((BC[negative_index, 0])**2+(BC[negative_index, 1])**2)**0.5)[:,None]*BA_dot_unit_BC[:, None]

                        BA_dot_unit_BC = BA_dot_BC / (
                            ((BC[:, 0]) ** 2 + (BC[:, 1]) ** 2) ** 0.5
                        )

                        BA_dot_unit_BC_times_unit_BC = (
                            BC
                            / (((BC[:, 0]) ** 2 + (BC[:, 1]) ** 2) ** 0.5)[:, None]
                            * BA_dot_unit_BC[:, None]
                        )

                        dx_distance[positive_index] = (
                            (
                                BA[positive_index, 0]
                                - BA_dot_unit_BC_times_unit_BC[positive_index, 0]
                            )
                            ** 2
                            + (
                                BA[positive_index, 1]
                                - BA_dot_unit_BC_times_unit_BC[positive_index, 1]
                            )
                            ** 2
                        ) ** 0.5
                        dx_distance[negative_index] = np.minimum(
                            (
                                (CA[negative_index, 0]) ** 2
                                + (CA[negative_index, 1]) ** 2
                            )
                            ** 0.5,
                            (
                                (BA[negative_index, 0]) ** 2
                                + (BA[negative_index, 1]) ** 2
                            )
                            ** 0.5,
                        )
                        if np.shape(zero_index)[0] != 0:
                            dx_distance[zero_index] = np.minimum(
                                ((CA[zero_index, 0]) ** 2 + (CA[zero_index, 1]) ** 2)
                                ** 0.5,
                                ((BA[zero_index, 0]) ** 2 + (BA[zero_index, 1]) ** 2)
                                ** 0.5,
                            )

                        min_distance = np.min(dx_distance)

                        min_index = np.argmin(dx_distance)

                        # H(d(x)), d(x) = ((x-x0)**2+(y-y0)**2)**0.5, need to find x0, y0
                        if (
                            min_index in positive_index
                        ):  # if the smallest distance is between AD, D in between BC
                            x_coor_min_point_segment = (
                                BA_dot_unit_BC_times_unit_BC[min_index, 0]
                                + BxByCxCy[min_index, 0]
                            )
                            y_coor_min_point_segment = (
                                BA_dot_unit_BC_times_unit_BC[min_index, 1]
                                + BxByCxCy[min_index, 1]
                            )
                        if (min_index in negative_index) or (
                            min_index in zero_index
                        ):  # if the smallest distance is AB or AC
                            if (
                                (CA[min_index, 0]) ** 2 + (CA[min_index, 1]) ** 2
                            ) ** 0.5 < (
                                (BA[min_index, 0]) ** 2 + (BA[min_index, 1]) ** 2
                            ) ** 0.5:
                                x_coor_min_point_segment = BxByCxCy[min_index, 2]
                                y_coor_min_point_segment = BxByCxCy[min_index, 3]
                            else:
                                x_coor_min_point_segment = BxByCxCy[min_index, 0]
                                y_coor_min_point_segment = BxByCxCy[min_index, 1]

                        heaviside_scaling_factor = 4.0e-7

                        heaviside = 1  # np.tanh(min_distance/heaviside_scaling_factor) # using initial structurednodal spacing to scale the distance to make sure the kernal drop down to zero when it is very close to the interface

                        # if j == node_ID_to_plot_1_index:
                        #     print('heaviside value node 1', heaviside)
                        # if j == node_ID_to_plot_2_index:
                        #     print('heaviside value node 2', heaviside)

                        phi_nonzero_index_row.append(i)
                        phi_nonzero_index_column.append(j)

                        heaviside_list.append(heaviside)
                        heaviside_plotting_points.append(x_G[i, :])

                        heaviside_P_x = 0  # (1-(np.tanh(min_distance/heaviside_scaling_factor))**2)*(x_G[i,0]-x_coor_min_point_segment)/min_distance/heaviside_scaling_factor
                        heaviside_P_y = 0  # (1-(np.tanh(min_distance/heaviside_scaling_factor))**2)*(x_G[i,1]-y_coor_min_point_segment)/min_distance/heaviside_scaling_factor

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

                else:
                    # if j == 855:
                    # print('yes')
                    phi_nonzerovalue_data.append(phi_ij)
                    phi_P_x_ij = phi_P_z * z_ij_P_x
                    phi_P_y_ij = phi_P_z * z_ij_P_y
                    phi_P_x_nonzerovalue_data.append(
                        phi_P_x_ij
                    )  # partial phi partial x
                    phi_P_y_nonzerovalue_data.append(
                        phi_P_y_ij
                    )  # partial phi partial y
                    z.append(z_ij)
                    z_P_x.append(z_ij_P_x)
                    z_P_y.append(z_ij_P_y)
                    phipz.append(phi_P_z)
                    phi_nonzero_index_row.append(i)
                    phi_nonzero_index_column.append(j)
                    for ii in range(3):
                        for jj in range(3):
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

        # print('M  ', M[0])

        # for ii in range(3):
        #     for jj in range(3):
        #         M[i][ii][jj] = M[i][ii][jj] + H[ii]*H_T[jj]*phi_ij
        #         M_P_x[i][ii][jj] = M_P_x[i][ii][jj] + H[ii]*H_T[jj]*phi_P_x_ij + H_P_x[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_x[jj]*phi_ij
        #         M_P_y[i][ii][jj] = M_P_y[i][ii][jj] + H[ii]*H_T[jj]*phi_P_y_ij + H_P_y[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_y[jj]*phi_ij
        # M_inv_P_x[i] = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_x[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
        # M_inv_P_y[i] = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_y[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
        # print()s

    return (
        phi_nonzero_index_row,
        phi_nonzero_index_column,
        phi_nonzerovalue_data,
        phi_P_x_nonzerovalue_data,
        phi_P_y_nonzerovalue_data,
        M,
        M_P_x,
        M_P_y,
        heaviside_list,
        heaviside_plotting_points,
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
):
    shape_func_value = []
    grad_shape_func_x_value = []
    grad_shape_func_y_value = []

    for ii in range(num_non_zero_phi_a):
        # print(ii)
        i = phi_nonzero_index_row[ii]
        j = phi_nonzero_index_column[ii]

        # compute the shape function and the gradient of shape function
        x_I = x_nodes[j]
        H_T = np.array(
            [1, (x_G[i][0] - x_I[0]), (x_G[i][1] - x_I[1])], dtype=np.float64
        )
        H = np.transpose(H_T)
        HT_P_x = np.array([0, 1, 0], dtype=np.float64)  # partial H partial x
        HT_P_y = np.array([0, 0, 1], dtype=np.float64)  # partial H partial y

        H_P_x = np.transpose(HT_P_x)
        H_P_y = np.transpose(HT_P_y)
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
            else:
                print("differential method is not defined")
        shape_func_value.append(shape_func_ij)
        grad_shape_func_x_value.append(grad_shape_func_x_ij)
        grad_shape_func_y_value.append(grad_shape_func_y_ij)

    return (
        shape_func_value,
        grad_shape_func_x_value,
        grad_shape_func_y_value,
    )


# def shape_grad_shape_func(x_G,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight,  H_P_x, H_P_y):

#     # matrix form:
#     HT0_m = np.tile(HT0, num_non_zero_phi_a).reshape(num_non_zero_phi_a, 3)

#     M_m = M[phi_nonzero_index_row]
#     H_T_0 = np.tile(np.array([1]), num_non_zero_phi_a)
#     H_T_1 = np.array(x_G[phi_nonzero_index_row, 0])-np.array(x_nodes[phi_nonzero_index_column, 0], dtype=np.float64)
#     H_T_2 = np.array(x_G[phi_nonzero_index_row, 1])-np.array(x_nodes[phi_nonzero_index_column, 1], dtype=np.float64)
#     H_m = np.array([H_T_0, H_T_1,H_T_2])
#     shape_func_value = ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist()
#     if differential_method =='implicite':
#         HT1_m = np.tile(HT1, num_non_zero_phi_a).reshape(num_non_zero_phi_a, 3)
#         HT2_m = np.tile(HT2, num_non_zero_phi_a).reshape(num_non_zero_phi_a, 3)
#         grad_shape_func_x_value = ((np.diagonal(np.dot(np.diagonal(np.dot(HT1_m, M_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist()
#         grad_shape_func_x_value = ((np.diagonal(np.dot(np.diagonal(np.dot(HT2_m, M_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist()
#     else:
#         if differential_method =='direct':
#             H_P_x_m = (np.repeat(H_P_x, num_non_zero_phi_a)).reshape(3,num_non_zero_phi_a)
#             H_P_y_m = (np.repeat(H_P_y, num_non_zero_phi_a)).reshape(3,num_non_zero_phi_a)
#             M_P_x_m = M_P_x[phi_nonzero_index_row]
#             M_P_y_m = M_P_y[phi_nonzero_index_row]
#             M_inv_P_x_m = -np.einsum('mns,msl->mnl', np.einsun('ijm,imk->ijk', np.linalg.inv(M), M_P_x_m), M_m)
#             M_inv_P_y_m = -np.einsum('mns,msl->mnl', np.einsun('ijm,imk->ijk', np.linalg.inv(M), M_P_y_m), M_m)
#             grad_shape_func_x_value = ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_P_x_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist() +\
#                                    ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_inv_P_x_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist() +\
#                                    ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_m), axis1=0,axis2 = 1).T, H_P_x_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist()
#             grad_shape_func_y_value = ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_P_y_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist() +\
#                                     ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_inv_P_y_m), axis1=0,axis2 = 1).T, H_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist() +\
#                                    ((np.diagonal(np.dot(np.diagonal(np.dot(HT0_m, M_m), axis1=0,axis2 = 1).T, H_P_y_m)).reshape(num_non_zero_phi_a,1))*((np.array(phi_nonzerovalue_data)).reshape(num_non_zero_phi_a,1))).tolist()

#         else:
#             print('differential method is not defined')


#     shape_func_times_det_J_time_weight_value = (shape_func_value*(np.asarray(det_J_time_weight)[phi_nonzero_index_row]).reshape(num_non_zero_phi_a,1)).tolist()
#     grad_shape_func_x_times_det_J_time_weight_value = (grad_shape_func_x_value*(np.asarray(det_J_time_weight)[phi_nonzero_index_row]).reshape(num_non_zero_phi_a,1)).tolist()
#     grad_shape_func_y_times_det_J_time_weight_value = (grad_shape_func_y_value*(np.asarray(det_J_time_weight)[phi_nonzero_index_row]).reshape(num_non_zero_phi_a,1)).tolist()

#     return shape_func_value, shape_func_times_det_J_time_weight_value, grad_shape_func_x_value, grad_shape_func_y_value, grad_shape_func_x_times_det_J_time_weight_value, grad_shape_func_y_times_det_J_time_weight_value
