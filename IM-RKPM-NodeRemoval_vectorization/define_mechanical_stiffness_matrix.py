import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm


def mechanical_C_tensor(
    num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle
):

    D_damage[D_damage > 0.9] = 0.9  # c = max(1-c_no_damage, 0.1)

    C11_ini = lambda_mechanical + 2 * mu  # c11 before rotation
    C22_ini = lambda_mechanical + 2 * mu
    C33_ini = mu
    C12_ini = lambda_mechanical
    C21_ini = C12_ini
    C13_ini = 0
    C31_ini = C13_ini
    C23_ini = 0
    C32_ini = C23_ini
    C_ini = np.array(
        [
            [C11_ini, C12_ini, C13_ini],
            [C21_ini, C22_ini, C23_ini],
            [C31_ini, C32_ini, C33_ini],
        ]
    )

    R11_c = np.cos(gauss_angle) ** 2
    R12_c = np.sin(gauss_angle) ** 2
    R13_c = 2 * np.sin(gauss_angle) * np.cos(gauss_angle)

    R21_c = np.sin(gauss_angle) ** 2
    R22_c = np.cos(gauss_angle) ** 2
    R23_c = -2 * np.sin(gauss_angle) * np.cos(gauss_angle)

    R31_c = -np.sin(gauss_angle) * np.cos(gauss_angle)
    R32_c = np.sin(gauss_angle) * np.cos(gauss_angle)
    R33_c = np.cos(gauss_angle) ** 2 - np.sin(gauss_angle) ** 2

    R_c = np.array(
        [[R11_c, R12_c, R13_c], [R21_c, R22_c, R23_c], [R31_c, R32_c, R33_c]]
    )

    # stiff tensor after rotation:

    C_mechanical = np.einsum(
        "imk,mjk->ijk",
        R_c,
        np.einsum("im,mjk->ijk", C_ini, np.transpose(R_c, (1, 0, 2))),
    )  # stiffness without damage

    C11 = C_mechanical[0, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C12 = C_mechanical[0, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C13 = C_mechanical[0, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C22 = C_mechanical[1, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C23 = C_mechanical[1, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C33 = C_mechanical[2, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    return C11, C12, C13, C22, C23, C33


def mechanical_stiffness_matrix(
    C11,
    C12,
    C13,
    C22,
    C23,
    C33,
    E,
    x_nodes,
    num_gauss_points_in_domain,
    grad_shape_func_x_times_det_J_time_weight,
    grad_shape_func_x,
    grad_shape_func_y_times_det_J_time_weight,
    grad_shape_func_y,
):

    C11 = C11.reshape(
        num_gauss_points_in_domain,
    )
    C12 = C12.reshape(
        num_gauss_points_in_domain,
    )
    C13 = C13.reshape(
        num_gauss_points_in_domain,
    )
    C22 = C22.reshape(
        num_gauss_points_in_domain,
    )
    C23 = C23.reshape(
        num_gauss_points_in_domain,
    )
    C33 = C33.reshape(
        num_gauss_points_in_domain,
    )

    K11_mechanical = (
        (sp.diags(C11).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_x
        + (sp.diags(C33).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_y
        + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_y
        + (sp.diags(C13).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_x
    )
    K12_mechanical = (
        (sp.diags(C12).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_y
        + (sp.diags(C33).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_x
        + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_x
        + (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_y
    )
    K21_mechanical = (
        (sp.diags(C12).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_x
        + (sp.diags(C33).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_y
        + (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_y
        + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_x
    )
    K22_mechanical = (
        (sp.diags(C22).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_y
        + (sp.diags(C33).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_x
        + (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T
        * grad_shape_func_x
        + (sp.diags(C23).dot(grad_shape_func_x_times_det_J_time_weight)).T
        * grad_shape_func_y
    )

    K_mechanical = bmat(
        [[K11_mechanical, K12_mechanical], [K21_mechanical, K22_mechanical]]
    )

    ### sp.diags(C22).dot(grad_shape_func_y_times_det_J_time_weight)).T is sparse matrix, (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x is dot product of two sparse matrixs
    # time1 = time.time()
    # # remove the rigid body motion
    # alpha = E/1000.0

    # # calculate the eigenvalue and eigenvectors of K matrix
    # EValue,EVector = eig(K_mechanical.toarray())
    # time2 = time.time()
    # print('eigen solving time: ', time2-time1)

    # index_zero_EValue = np.argsort(EValue)[:3] # the index of the three zero eigen values in EValue,

    # EVector1 = (np.array(EVector[:,index_zero_EValue[0]]).reshape(2*np.shape(x_nodes)[0],1))
    # EVector2 = (np.array(EVector[:,index_zero_EValue[1]]).reshape(2*np.shape(x_nodes)[0],1))
    # EVector3 = (np.array(EVector[:,index_zero_EValue[2]]).reshape(2*np.shape(x_nodes)[0],1))   # three eigenvectors corresponding to three zero eigenvalues

    # K_mechanical = csc_matrix(K_mechanical+alpha*(np.dot(EVector1, EVector1.T)+np.dot(EVector2, EVector2.T)+np.dot(EVector3, EVector3.T)))

    return K_mechanical


def mechanical_force_matrix(
    x_G,
    C11,
    C12,
    C13,
    C22,
    C23,
    C33,
    epsilon_D1,
    epsilon_D2,
    epsilon_D3,
    grad_shape_func_x_times_det_J_time_weight,
    grad_shape_func_y_times_det_J_time_weight,
):
    num_gauss_points = np.shape(x_G)[0]

    c_e_D1 = C11 * epsilon_D1 + C12 * epsilon_D2 + C13 * epsilon_D3
    c_e_D2 = C12 * epsilon_D1 + C22 * epsilon_D2 + C23 * epsilon_D3
    c_e_D3 = C13 * epsilon_D1 + C23 * epsilon_D2 + C33 * epsilon_D3

    # assemble the force matrix for mechanical simulation
    f1_mechanical = (
        grad_shape_func_x_times_det_J_time_weight.T * c_e_D1
        + grad_shape_func_y_times_det_J_time_weight.T * c_e_D3
    )
    f2_mechanical = (
        grad_shape_func_y_times_det_J_time_weight.T * c_e_D2
        + grad_shape_func_x_times_det_J_time_weight.T * c_e_D3
    )
    f_mechanical = np.concatenate((f1_mechanical, f2_mechanical))
    # f_mechanical = f_mechanical[reorder_index, :]

    return f_mechanical
