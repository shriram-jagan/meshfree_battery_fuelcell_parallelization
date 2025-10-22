import time

start_time = time.time()
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm


def diffusion_matrix(
    shape_func,
    shape_func_times_det_J_time_weight,
    grad_shape_func_x,
    D_R11,
    grad_shape_func_y,
    D_R21,
    grad_shape_func_x_times_det_J_time_weight,
    D_R12,
    D_R22,
    grad_shape_func_y_times_det_J_time_weight,
    dD_dc_R11,
    dD_dc_R21,
    c_n1,
    dD_dc_R12,
    dD_dc_R22,
    shape_func_b,
    djbv_deta,
    dE_eq_dc,
    shape_func_b_times_det_J_b_time_weight,
    djbv_dj0,
    dj0_dc,
    j_BV,
    j_applied,
    x_G_b,
    k_con,
    dt,
    Fday,
    c_n,
    phi_n1,
):

    M_matrix = shape_func.T * shape_func_times_det_J_time_weight  # sparse matrix
    K_cc = (
        (grad_shape_func_x).multiply(D_R11) + (grad_shape_func_y).multiply(D_R21)
    ).T * grad_shape_func_x_times_det_J_time_weight + (
        (grad_shape_func_x).multiply(D_R12) + (grad_shape_func_y).multiply(D_R22)
    ).T * grad_shape_func_y_times_det_J_time_weight  # sparse matrix

    K_cc_D = (
        (
            (grad_shape_func_x).multiply(dD_dc_R11)
            + (grad_shape_func_y).multiply(dD_dc_R21)
        ).multiply(grad_shape_func_x * c_n1)
        + (
            (grad_shape_func_x).multiply(dD_dc_R12)
            + (grad_shape_func_y).multiply(dD_dc_R22)
        ).multiply(grad_shape_func_y * c_n1)
    ).T * shape_func_times_det_J_time_weight  # sparse matrix

    K_cp_Eeq = (
        shape_func_b.multiply((djbv_deta * dE_eq_dc))
    ).T * shape_func_b_times_det_J_b_time_weight  # sparse matrix
    K_cp_j0 = (
        shape_func_b.multiply((-djbv_dj0 * dj0_dc))
    ).T * shape_func_b_times_det_J_b_time_weight  # sparse matrix
    K_cp_BV = (
        shape_func_b.multiply((-djbv_deta))
    ).T * shape_func_b_times_det_J_b_time_weight  # sparse matrix
    f_c = shape_func_b_times_det_J_b_time_weight.T * (-j_BV)  # array
    f_phi = shape_func_b_times_det_J_b_time_weight.T * (
        -(j_BV + j_applied * np.array(np.ones((np.shape(x_G_b)[0], 1))))
    )  # array
    K_phiphi = (
        grad_shape_func_x.T * grad_shape_func_x_times_det_J_time_weight
        + grad_shape_func_y.T * grad_shape_func_y_times_det_J_time_weight
    ).multiply(
        k_con
    )  # sparse

    K11 = (
        M_matrix
        + K_cc.multiply(dt)
        - (K_cp_j0 + K_cp_Eeq).multiply(dt / Fday)
        + K_cc_D.multiply(dt)
    )
    K12 = K_cp_BV.multiply(-dt / Fday)
    K21 = -K_cp_Eeq - K_cp_j0
    K22 = K_phiphi - K_cp_BV

    f1 = f_c * dt / Fday - (K_cc * c_n1) * dt - M_matrix * (c_n1 - c_n)
    f2 = f_phi - K_phiphi * phi_n1

    K = bmat([[K11, K12], [K21, K22]])
    f = np.concatenate((f1, f2))

    return K, f
