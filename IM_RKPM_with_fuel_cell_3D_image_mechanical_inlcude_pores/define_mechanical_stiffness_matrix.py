import time
import numpy as np
from numpy import sign

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit
import scipy.sparse as sp

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

def mechanical_C_tensor(num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle):

    D_damage[D_damage>0.9] = 0.9    #c = max(1-c_no_damage, 0.1)

    C11_ini = lambda_mechanical+2*mu           # c11 before rotation
    C22_ini = lambda_mechanical+2*mu               
    C33_ini = mu
    C12_ini = lambda_mechanical
    C21_ini = C12_ini
    if len(np.shape(lambda_mechanical)) == 0:
        C13_ini = 0
        C31_ini = C13_ini
        C23_ini = 0
        C32_ini = C23_ini
    else:
        C13_ini = np.zeros(num_gauss_points_in_domain)
        C31_ini = C13_ini
        C23_ini = np.zeros(num_gauss_points_in_domain)
        C32_ini = C23_ini
    C_ini = np.array([[C11_ini, C12_ini, C13_ini],[C21_ini, C22_ini, C23_ini],[C31_ini, C32_ini, C33_ini]])

    
    R11_c = np.cos(gauss_angle)**2
    R12_c = np.sin(gauss_angle)**2
    R13_c = 2*np.sin(gauss_angle)*np.cos(gauss_angle)

    R21_c = np.sin(gauss_angle)**2
    R22_c = np.cos(gauss_angle)**2
    R23_c = -2*np.sin(gauss_angle)*np.cos(gauss_angle)

    R31_c = -np.sin(gauss_angle)*np.cos(gauss_angle)
    R32_c = np.sin(gauss_angle)*np.cos(gauss_angle)
    R33_c = np.cos(gauss_angle)**2-np.sin(gauss_angle)**2

    R_c = np.array([[R11_c, R12_c, R13_c],[R21_c, R22_c, R23_c],[R31_c, R32_c, R33_c]])

    # stiff tensor after rotation:
    
    if len(np.shape(C11_ini)) == 0: # C11_ini is a float not array
        C_mechanical = np.einsum('imk,mjk->ijk', R_c, np.einsum('im,mjk->ijk', C_ini, np.transpose(R_c, (1,0,2)))) # stiffness without damage
    else: # C11_ini is array, different material properties at different gauss points.
        C_mechanical = np.einsum('imk,mjk->ijk', R_c, np.einsum('imk,mjk->ijk', C_ini, np.transpose(R_c, (1,0,2)))) # stiffness without damage


    C11 = C_mechanical[0,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C12 = C_mechanical[0,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C13 = C_mechanical[0,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C22 = C_mechanical[1,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C23 = C_mechanical[1,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C33 = C_mechanical[2,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    return C11, C12, C13, C22, C23, C33

def mechanical_C_tensor_3d(num_gauss_points_in_domain, D_damage, lambda_mechanical, mu, gauss_angle, gauss_rotation_axis):

    D_damage[D_damage>0.9] = 0.9    #c = max(1-c_no_damage, 0.1)

    C11_ini = lambda_mechanical+2*mu           # c11 before rotation
    C22_ini = lambda_mechanical+2*mu   
    C33_ini = lambda_mechanical+2*mu             
    C44_ini = mu
    C55_ini = mu
    C66_ini = mu
    C12_ini = lambda_mechanical
    C13_ini = lambda_mechanical
    C23_ini = lambda_mechanical
    C21_ini = C12_ini
    C31_ini = C13_ini
    C32_ini = C23_ini
    
    if len(np.shape(lambda_mechanical)) == 0:
        C14_ini = 0
        C15_ini = 0
        C16_ini = 0
        
        
        C24_ini = 0
        C25_ini = 0
        C26_ini = 0
        
        
        C34_ini = 0
        C35_ini = 0
        C36_ini = 0

        C41_ini = 0
        C42_ini = 0
        C43_ini = 0
        C45_ini = 0
        C46_ini = 0

        C51_ini = 0
        C52_ini = 0
        C53_ini = 0
        C54_ini = 0
        C56_ini = 0

        C61_ini = 0
        C62_ini = 0
        C63_ini = 0
        C64_ini = 0
        C65_ini = 0

    else:
        C14_ini = np.zeros(num_gauss_points_in_domain)
        C15_ini = np.zeros(num_gauss_points_in_domain)
        C16_ini = np.zeros(num_gauss_points_in_domain)
        
        
        C24_ini = np.zeros(num_gauss_points_in_domain)
        C25_ini = np.zeros(num_gauss_points_in_domain)
        C26_ini = np.zeros(num_gauss_points_in_domain)
        
        
        C34_ini = np.zeros(num_gauss_points_in_domain)
        C35_ini = np.zeros(num_gauss_points_in_domain)
        C36_ini = np.zeros(num_gauss_points_in_domain)

        C41_ini = np.zeros(num_gauss_points_in_domain)
        C42_ini = np.zeros(num_gauss_points_in_domain)
        C43_ini = np.zeros(num_gauss_points_in_domain)
        C45_ini = np.zeros(num_gauss_points_in_domain)
        C46_ini = np.zeros(num_gauss_points_in_domain)

        C51_ini = np.zeros(num_gauss_points_in_domain)
        C52_ini = np.zeros(num_gauss_points_in_domain)
        C53_ini = np.zeros(num_gauss_points_in_domain)
        C54_ini = np.zeros(num_gauss_points_in_domain)
        C56_ini = np.zeros(num_gauss_points_in_domain)

        C61_ini = np.zeros(num_gauss_points_in_domain)
        C62_ini = np.zeros(num_gauss_points_in_domain)
        C63_ini = np.zeros(num_gauss_points_in_domain)
        C64_ini = np.zeros(num_gauss_points_in_domain)
        C65_ini = np.zeros(num_gauss_points_in_domain)

        C_ini = np.array([[C11_ini, C12_ini, C13_ini, C14_ini, C15_ini, C16_ini], \
                          [C21_ini, C22_ini, C23_ini, C24_ini, C25_ini, C26_ini], \
                          [C31_ini, C32_ini, C33_ini, C34_ini, C35_ini, C36_ini], \
                          [C41_ini, C42_ini, C43_ini, C44_ini, C45_ini, C46_ini], \
                          [C51_ini, C52_ini, C53_ini, C54_ini, C55_ini, C56_ini], \
                          [C61_ini, C62_ini, C63_ini, C64_ini, C65_ini, C66_ini], \
                            ])

    R11_c = np.cos(gauss_angle) + gauss_rotation_axis[:, 0]**2*(1-np.cos(gauss_angle))
    R12_c = gauss_rotation_axis[:, 0]*gauss_rotation_axis[:, 1]*(1-np.cos(gauss_angle))-gauss_rotation_axis[:, 2]*np.sin(gauss_angle)
    R13_c = gauss_rotation_axis[:, 0]*gauss_rotation_axis[:, 2]*(1-np.cos(gauss_angle))+gauss_rotation_axis[:, 1]*np.sin(gauss_angle)

    R21_c = gauss_rotation_axis[:, 0]*gauss_rotation_axis[:, 1]*(1-np.cos(gauss_angle))+gauss_rotation_axis[:, 2]*np.sin(gauss_angle)
    R22_c = np.cos(gauss_angle) + gauss_rotation_axis[:, 1]**2*(1-np.cos(gauss_angle))
    R23_c = gauss_rotation_axis[:, 2]*gauss_rotation_axis[:, 1]*(1-np.cos(gauss_angle))-gauss_rotation_axis[:, 0]*np.sin(gauss_angle)

    R31_c = gauss_rotation_axis[:, 2]*gauss_rotation_axis[:, 0]*(1-np.cos(gauss_angle))-gauss_rotation_axis[:, 1]*np.sin(gauss_angle)
    R32_c = gauss_rotation_axis[:, 2]*gauss_rotation_axis[:, 1]*(1-np.cos(gauss_angle))+gauss_rotation_axis[:, 0]*np.sin(gauss_angle)
    R33_c = np.cos(gauss_angle) + gauss_rotation_axis[:, 2]**2*(1-np.cos(gauss_angle))

    T11_c = R11_c**2
    T12_c = R12_c**2
    T13_c = R13_c**2
    T14_c = 2*R12_c*R13_c
    T15_c = 2*R11_c*R13_c
    T16_c = 2*R11_c*R12_c

    T21_c = R21_c**2
    T22_c = R22_c**2
    T23_c = R23_c**2
    T24_c = 2*R22_c*R23_c
    T25_c = 2*R21_c*R23_c
    T26_c = 2*R21_c*R22_c

    T31_c = R31_c**2
    T32_c = R32_c**2
    T33_c = R33_c**2
    T34_c = 2*R32_c*R33_c
    T35_c = 2*R31_c*R33_c
    T36_c = 2*R31_c*R32_c

    T41_c = R21_c*R31_c
    T42_c = R22_c*R32_c
    T43_c = R23_c*R33_c
    T44_c = R22_c*R33_c+R23_c*R32_c
    T45_c = R21_c*R33_c+R23_c*R31_c
    T46_c = R21_c*R32_c+R22_c*R31_c

    T51_c = R11_c*R31_c
    T52_c = R12_c*R32_c
    T53_c = R13_c*R33_c
    T54_c = R12_c*R33_c+R13_c*R32_c
    T55_c = R11_c*R33_c+R13_c*R31_c
    T56_c = R11_c*R32_c+R12_c*R31_c

    T61_c = R11_c*R21_c
    T62_c = R12_c*R22_c
    T63_c = R13_c*R23_c
    T64_c = R12_c*R23_c+R13_c*R22_c
    T65_c = R11_c*R23_c+R13_c*R21_c
    T66_c = R11_c*R22_c+R12_c*R21_c

    T_c = np.array([[T11_c, T12_c, T13_c, T14_c, T15_c, T16_c],\
                    [T21_c, T22_c, T23_c, T24_c, T25_c, T26_c],\
                    [T31_c, T32_c, T33_c, T34_c, T35_c, T36_c],\
                    [T41_c, T42_c, T43_c, T44_c, T45_c, T46_c],\
                    [T51_c, T52_c, T53_c, T54_c, T55_c, T56_c],\
                    [T61_c, T62_c, T63_c, T64_c, T65_c, T66_c],\
                        ])

    # stiff tensor after rotation:
    if len(np.shape(lambda_mechanical)) == 0:
        C_mechanical = np.einsum('imk,mjk->ijk', T_c, np.einsum('im,mjk->ijk', C_ini, np.transpose(T_c, (1,0,2)))) # stiffness without damage
    else:
        C_mechanical = np.einsum('imk,mjk->ijk', T_c, np.einsum('imk,mjk->ijk', C_ini, np.transpose(T_c, (1,0,2)))) # stiffness without damage


    C11 = C_mechanical[0,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C12 = C_mechanical[0,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C13 = C_mechanical[0,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C14 = C_mechanical[0,3].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C15 = C_mechanical[0,4].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C16 = C_mechanical[0,5].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    C21 = C_mechanical[1,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C22 = C_mechanical[1,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C23 = C_mechanical[1,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C24 = C_mechanical[1,3].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C25 = C_mechanical[1,4].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C26 = C_mechanical[1,5].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    C31 = C_mechanical[2,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C32 = C_mechanical[2,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C33 = C_mechanical[2,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C34 = C_mechanical[2,3].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C35 = C_mechanical[2,4].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C36 = C_mechanical[2,5].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    C41 = C_mechanical[3,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C42 = C_mechanical[3,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C43 = C_mechanical[3,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C44 = C_mechanical[3,3].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C45 = C_mechanical[3,4].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C46 = C_mechanical[3,5].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    C51 = C_mechanical[4,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C52 = C_mechanical[4,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C53 = C_mechanical[4,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C54 = C_mechanical[4,3].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C55 = C_mechanical[4,4].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C56 = C_mechanical[4,5].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    C61 = C_mechanical[5,0].reshape(num_gauss_points_in_domain,1)*(1-D_damage) # shape: number of gauss points
    C62 = C_mechanical[5,1].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C63 = C_mechanical[5,2].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C64 = C_mechanical[5,3].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C65 = C_mechanical[5,4].reshape(num_gauss_points_in_domain,1)*(1-D_damage)
    C66 = C_mechanical[5,5].reshape(num_gauss_points_in_domain,1)*(1-D_damage)

    C = np.array([[C11, C12, C13, C14, C15, C16], \
                          [C21, C22, C23, C24, C25, C26], \
                          [C31, C32, C33, C34, C35, C36], \
                          [C41, C42, C43, C44, C45, C46], \
                          [C51, C52, C53, C54, C55, C56], \
                          [C61, C62, C63, C64, C65, C66], \
                            ])

    return C, T_c

def mechanical_stiffness_matrix_fuel_cell(C11, C12, C13, C22, C23, C33, num_gauss_points_in_domain, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y_times_det_J_time_weight, grad_shape_func_y, beta_Nitsche,\
                                shape_func_b_electrolyte, shape_func_b_times_det_J_b_time_weight_electrolyte, \
                                    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte,\
                                        normal_vector_x_electrolyte, normal_vector_y_electrolyte, \
                                            shape_func_fixed_point, grad_shape_func_x_fixed_point, grad_shape_func_y_fixed_point):
    
    num_gauss_points_on_boundary = np.shape(shape_func_b_electrolyte.toarray())[0]
    num_fixed_point = np.shape(shape_func_fixed_point.toarray())[0]
    
    C11 = C11.reshape(num_gauss_points_in_domain,)
    C12 = C12.reshape(num_gauss_points_in_domain,)
    C13 = C13.reshape(num_gauss_points_in_domain,)
    C22 = C22.reshape(num_gauss_points_in_domain,)
    C23 = C23.reshape(num_gauss_points_in_domain,)
    C33 = C33.reshape(num_gauss_points_in_domain,)

    K11_mechanical_domain_int = (sp.diags(C11).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C33).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y \
            + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y + (sp.diags(C13).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x
    K12_mechanical_domain_int = (sp.diags(C12).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+(sp.diags(C33).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x \
                + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y
    K21_mechanical_domain_int = (sp.diags(C12).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C33).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y \
                + (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+(sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x
    K22_mechanical_domain_int = (sp.diags(C22).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+(sp.diags(C33).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x \
                +(sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C23).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y
    
    # if fixed alogn x at left boundary and fixed xy at corner
    # K11_mechanical_boundary_int = (shape_func_b_electrolyte.multiply(beta_Nitsche)).T*shape_func_b_times_det_J_b_time_weight_electrolyte  \
    # - (sp.diags(C11[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    # - (sp.diags(C33[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) 
    
    # K11_mechanical_point_int = (shape_func_fixed_point.multiply(beta_Nitsche)).T*shape_func_fixed_point  \
    # - (sp.diags(C11[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C13[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C13[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    # - (sp.diags(C33[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) 
    
    # K12_mechanical_point_int =  \
    # - (sp.diags(C13[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C33[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C12[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    # - (sp.diags(C23[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) 
    
    # K21_mechanical_boundary_int = \
    # - (sp.diags(C12[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C23[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    # - (sp.diags(C33[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) 
    
    # K21_mechanical_point_int =  \
    # - (sp.diags(C12[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C13[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C23[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    # - (sp.diags(C33[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) 
    
    # K22_mechanical_point_int = (shape_func_fixed_point.multiply(beta_Nitsche)).T*shape_func_fixed_point  \
    # - (sp.diags(C23[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C33[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    # - (sp.diags(C22[:num_fixed_point]).dot(grad_shape_func_y_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    # - (sp.diags(C23[:num_fixed_point]).dot(grad_shape_func_x_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) 
    
    # K11_mechanical = K11_mechanical_domain_int + K11_mechanical_boundary_int + K11_mechanical_point_int
    # K12_mechanical = K12_mechanical_domain_int + K12_mechanical_point_int
    # K21_mechanical = K21_mechanical_domain_int + K21_mechanical_boundary_int + K21_mechanical_point_int
    # K22_mechanical = K22_mechanical_domain_int + K22_mechanical_point_int
    
    # if fixed both x y directions at left boundary
    K11_mechanical_boundary_int = (shape_func_b_electrolyte.multiply(beta_Nitsche)).T*shape_func_b_times_det_J_b_time_weight_electrolyte  \
    - (sp.diags(C11[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C33[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) 
    
    K12_mechanical_boundary_int =  \
    - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C33[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C12[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C23[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) 
    
    K21_mechanical_boundary_int = \
    - (sp.diags(C12[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C13[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C23[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C33[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) 
    
    K22_mechanical_boundary_int = (shape_func_b_electrolyte.multiply(beta_Nitsche)).T*shape_func_b_times_det_J_b_time_weight_electrolyte \
    - (sp.diags(C23[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C33[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C22[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C23[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) 
    
    
    K11_mechanical = K11_mechanical_domain_int + K11_mechanical_boundary_int 
    K12_mechanical = K12_mechanical_domain_int + K12_mechanical_boundary_int
    K21_mechanical = K21_mechanical_domain_int + K21_mechanical_boundary_int 
    K22_mechanical = K22_mechanical_domain_int + K22_mechanical_boundary_int

    K_mechanical = bmat([[K11_mechanical, K12_mechanical], [K21_mechanical, K22_mechanical]]) 

    
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

def mechanical_stiffness_matrix_3d_fuel_cell(C,num_gauss_points_in_domain, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y_times_det_J_time_weight, grad_shape_func_y, grad_shape_func_z_times_det_J_time_weight, grad_shape_func_z, \
                                   beta_Nitsche, \
                                    shape_func_fixed_point, shape_func_times_det_J_time_weight_fixed_point,\
                                    grad_shape_func_x_fixed_point, grad_shape_func_x_times_det_J_time_weight_fixed_point,\
                                    grad_shape_func_y_fixed_point, grad_shape_func_y_times_det_J_time_weight_fixed_point,\
                                    grad_shape_func_z_fixed_point, grad_shape_func_z_times_det_J_time_weight_fixed_point,\
                                    normal_vector_x_electrolyte, normal_vector_y_electrolyte,normal_vector_z_electrolyte,\
                                    shape_func_b_electrolyte, shape_func_b_times_det_J_b_time_weight_electrolyte,\
                                    grad_shape_func_b_x_electrolyte, grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte,\
                                    grad_shape_func_b_y_electrolyte, grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte,\
                                    grad_shape_func_b_z_electrolyte, grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte):
    
    num_gauss_points_on_boundary = np.shape(shape_func_b_electrolyte.toarray())[0]
    num_gauss_points_on_fixed_line = np.shape(shape_func_fixed_point.toarray())[0]
    
    C11 = C[0,0].reshape(num_gauss_points_in_domain,)
    C12 = C[0,1].reshape(num_gauss_points_in_domain,)
    C13 = C[0,2].reshape(num_gauss_points_in_domain,)
    C14 = C[0,3].reshape(num_gauss_points_in_domain,)
    C15 = C[0,4].reshape(num_gauss_points_in_domain,)
    C16 = C[0,5].reshape(num_gauss_points_in_domain,)

    C21 = C[1,0].reshape(num_gauss_points_in_domain,)
    C22 = C[1,1].reshape(num_gauss_points_in_domain,)
    C23 = C[1,2].reshape(num_gauss_points_in_domain,)
    C24 = C[1,3].reshape(num_gauss_points_in_domain,)
    C25 = C[1,4].reshape(num_gauss_points_in_domain,)
    C26 = C[1,5].reshape(num_gauss_points_in_domain,)

    C31 = C[2,0].reshape(num_gauss_points_in_domain,)
    C32 = C[2,1].reshape(num_gauss_points_in_domain,)
    C33 = C[2,2].reshape(num_gauss_points_in_domain,)
    C34 = C[2,3].reshape(num_gauss_points_in_domain,)
    C35 = C[2,4].reshape(num_gauss_points_in_domain,)
    C36 = C[2,5].reshape(num_gauss_points_in_domain,)

    C41 = C[3,0].reshape(num_gauss_points_in_domain,)
    C42 = C[3,1].reshape(num_gauss_points_in_domain,)
    C43 = C[3,2].reshape(num_gauss_points_in_domain,)
    C44 = C[3,3].reshape(num_gauss_points_in_domain,)
    C45 = C[3,4].reshape(num_gauss_points_in_domain,)
    C46 = C[3,5].reshape(num_gauss_points_in_domain,)

    C51 = C[4,0].reshape(num_gauss_points_in_domain,)
    C52 = C[4,1].reshape(num_gauss_points_in_domain,)
    C53 = C[4,2].reshape(num_gauss_points_in_domain,)
    C54 = C[4,3].reshape(num_gauss_points_in_domain,)
    C55 = C[4,4].reshape(num_gauss_points_in_domain,)
    C56 = C[4,5].reshape(num_gauss_points_in_domain,)

    C61 = C[5,0].reshape(num_gauss_points_in_domain,)
    C62 = C[5,1].reshape(num_gauss_points_in_domain,)
    C63 = C[5,2].reshape(num_gauss_points_in_domain,)
    C64 = C[5,3].reshape(num_gauss_points_in_domain,)
    C65 = C[5,4].reshape(num_gauss_points_in_domain,)
    C66 = C[5,5].reshape(num_gauss_points_in_domain,)

    K11_mechanical_doamin_int = (sp.diags(C11).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C14).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C15).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C41).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C44).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C45).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C51).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C54).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C55).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z
    
    K12_mechanical_doamin_int = (sp.diags(C12).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C14).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C16).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C42).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C44).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C46).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C52).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C54).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C56).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z
    
    K13_mechanical_doamin_int = (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C15).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C16).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C43).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C45).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C46).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C53).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C55).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C56).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y
    
    K21_mechanical_doamin_int = (sp.diags(C21).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C24).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C25).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C41).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C44).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C45).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C61).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C64).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C65).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z
    
    K22_mechanical_doamin_int = (sp.diags(C22).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C24).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C26).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C42).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C44).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C46).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C62).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C64).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C66).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z
    
    K23_mechanical_doamin_int = (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C25).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C26).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C43).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C45).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C46).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C63).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C65).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C66).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y
    
    K31_mechanical_doamin_int = (sp.diags(C31).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C34).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C35).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C51).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C54).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C55).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C61).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C64).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C65).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z
    
    K32_mechanical_doamin_int = (sp.diags(C32).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C34).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C36).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C52).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C54).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C56).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C62).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C64).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C66).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z
    
    K33_mechanical_doamin_int = (sp.diags(C33).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C35).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C36).dot(grad_shape_func_z_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C53).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C55).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C56).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+ \
                     (sp.diags(C63).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_z+ \
                     (sp.diags(C65).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+ \
                     (sp.diags(C66).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y
    
    K11_mechanical_line_int = (shape_func_times_det_J_time_weight_fixed_point.multiply(beta_Nitsche)).T*shape_func_fixed_point\
    - (sp.diags(C11[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C14[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C15[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C41[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C44[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C45[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C51[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C54[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C55[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
   
    K12_mechanical_line_int = \
    - (sp.diags(C14[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C44[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C45[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C21[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C24[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C25[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C61[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C64[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C65[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K12_mechanical_boundary_int = \
    - (sp.diags(C14[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C44[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C45[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C21[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C24[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C25[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C61[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C64[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C65[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
   
    K13_mechanical_line_int = \
    - (sp.diags(C51[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C54[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C55[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C61[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C64[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C65[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C31[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C34[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C35[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K21_mechanical_line_int = \
    - (sp.diags(C12[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C14[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C16[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C42[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C44[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C46[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C52[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C54[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C56[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K22_mechanical_line_int = (shape_func_times_det_J_time_weight_fixed_point.multiply(beta_Nitsche)).T*shape_func_fixed_point\
    - (sp.diags(C42[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C44[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C46[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C22[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C24[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C26[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C62[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C64[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C66[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K22_mechanical_boundary_int = (shape_func_b_times_det_J_b_time_weight_electrolyte.multiply(beta_Nitsche)).T*shape_func_b_electrolyte\
    - (sp.diags(C42[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C44[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C46[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C22[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C24[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C26[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C62[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C64[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C66[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
   
    K23_mechanical_line_int = \
    - (sp.diags(C52[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C54[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C56[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C62[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C64[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C66[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C32[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C34[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C36[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K31_mechanical_line_int = \
    - (sp.diags(C13[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C15[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C16[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C43[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C45[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C46[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C53[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C55[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C56[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K32_mechanical_line_int = \
    - (sp.diags(C43[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C45[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C46[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C23[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C25[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C26[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C63[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C65[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C66[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K32_mechanical_boundary_int = \
    - (sp.diags(C43[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C45[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C46[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C23[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C25[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C26[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C63[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C65[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C66[:num_gauss_points_on_boundary]).dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)).T*(shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)) \
   
    K33_mechanical_line_int = (shape_func_times_det_J_time_weight_fixed_point.multiply(beta_Nitsche)).T*shape_func_fixed_point\
    - (sp.diags(C53[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C55[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C56[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) \
    - (sp.diags(C63[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C65[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C66[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) \
    - (sp.diags(C33[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C35[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) \
    - (sp.diags(C36[:num_gauss_points_on_fixed_line]).dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)).T*(shape_func_fixed_point.multiply(normal_vector_z_electrolyte)) 
    
    K11_mechanical = K11_mechanical_doamin_int + K11_mechanical_line_int
    K12_mechanical = K12_mechanical_doamin_int + K12_mechanical_boundary_int + K12_mechanical_line_int
    K13_mechanical = K13_mechanical_doamin_int + K13_mechanical_line_int
    K21_mechanical = K21_mechanical_doamin_int + K21_mechanical_line_int
    K22_mechanical = K22_mechanical_doamin_int + K22_mechanical_boundary_int + K22_mechanical_line_int
    K23_mechanical = K23_mechanical_doamin_int + K23_mechanical_line_int
    K31_mechanical = K31_mechanical_doamin_int + K31_mechanical_line_int
    K32_mechanical = K32_mechanical_doamin_int + K32_mechanical_boundary_int + K32_mechanical_line_int
    K33_mechanical = K33_mechanical_doamin_int + K33_mechanical_line_int
    
    K_mechanical = bmat([[K11_mechanical, K12_mechanical, K13_mechanical], [K21_mechanical, K22_mechanical, K23_mechanical], [K31_mechanical, K32_mechanical, K33_mechanical]]) 

    
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

def mechanical_stiffness_matrix_battery(C11, C12, C13, C22, C23, C33, num_gauss_points_in_domain, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y_times_det_J_time_weight, grad_shape_func_y):

    C11 = C11.reshape(num_gauss_points_in_domain,)
    C12 = C12.reshape(num_gauss_points_in_domain,)
    C13 = C13.reshape(num_gauss_points_in_domain,)
    C22 = C22.reshape(num_gauss_points_in_domain,)
    C23 = C23.reshape(num_gauss_points_in_domain,)
    C33 = C33.reshape(num_gauss_points_in_domain,)

    K11_mechanical = (sp.diags(C11).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C33).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y \
            + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y + (sp.diags(C13).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x
    K12_mechanical = (sp.diags(C12).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y+(sp.diags(C33).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x \
                + (sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y
    K21_mechanical = (sp.diags(C12).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C33).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y \
                + (sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+(sp.diags(C13).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x
    K22_mechanical = (sp.diags(C22).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_y+(sp.diags(C33).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_x \
                +(sp.diags(C23).dot(grad_shape_func_y_times_det_J_time_weight)).T*grad_shape_func_x+(sp.diags(C23).dot(grad_shape_func_x_times_det_J_time_weight)).T*grad_shape_func_y
    
    K_mechanical = bmat([[K11_mechanical, K12_mechanical], [K21_mechanical, K22_mechanical]]) 

    
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


def mechanical_force_matrix(x_G, C11, C12, C13, C22, C23, C33, epsilon_D1, epsilon_D2, epsilon_D3, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight):
    num_gauss_points = np.shape(x_G)[0]

    c_e_D1 = C11*epsilon_D1+C12*epsilon_D2+C13*epsilon_D3
    c_e_D2 = C12*epsilon_D1+C22*epsilon_D2+C23*epsilon_D3
    c_e_D3 = C13*epsilon_D1+C23*epsilon_D2+C33*epsilon_D3

    # assemble the force matrix for mechanical simulation
    f1_mechanical = grad_shape_func_x_times_det_J_time_weight.T*c_e_D1+grad_shape_func_y_times_det_J_time_weight.T*c_e_D3
    f2_mechanical = grad_shape_func_y_times_det_J_time_weight.T*c_e_D2+grad_shape_func_x_times_det_J_time_weight.T*c_e_D3
    f_mechanical = np.concatenate((f1_mechanical, f2_mechanical))

    return f_mechanical







def mechanical_force_matrix_3d(x_G, C, epsilon_D1, epsilon_D2, epsilon_D3, epsilon_D4, epsilon_D5, epsilon_D6, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight, grad_shape_func_z_times_det_J_time_weight):
    num_gauss_points = np.shape(x_G)[0]

    c_e_D1 = C[0,0]*epsilon_D1+C[0,1]*epsilon_D2+C[0,2]*epsilon_D3+C[0,3]*epsilon_D4+C[0,4]*epsilon_D5+C[0,5]*epsilon_D6
    c_e_D2 = C[1,0]*epsilon_D1+C[1,1]*epsilon_D2+C[1,2]*epsilon_D3+C[1,3]*epsilon_D4+C[1,4]*epsilon_D5+C[1,5]*epsilon_D6
    c_e_D3 = C[2,0]*epsilon_D1+C[2,1]*epsilon_D2+C[2,2]*epsilon_D3+C[2,3]*epsilon_D4+C[2,4]*epsilon_D5+C[2,5]*epsilon_D6
    c_e_D4 = C[3,0]*epsilon_D1+C[3,1]*epsilon_D2+C[3,2]*epsilon_D3+C[3,3]*epsilon_D4+C[3,4]*epsilon_D5+C[3,5]*epsilon_D6
    c_e_D5 = C[4,0]*epsilon_D1+C[4,1]*epsilon_D2+C[4,2]*epsilon_D3+C[4,3]*epsilon_D4+C[4,4]*epsilon_D5+C[4,5]*epsilon_D6
    c_e_D6 = C[5,0]*epsilon_D1+C[5,1]*epsilon_D2+C[5,2]*epsilon_D3+C[5,3]*epsilon_D4+C[5,4]*epsilon_D5+C[5,5]*epsilon_D6

    # assemble the force matrix for mechanical simulation
    f1_mechanical = grad_shape_func_x_times_det_J_time_weight.T*c_e_D1+grad_shape_func_y_times_det_J_time_weight.T*c_e_D4+grad_shape_func_z_times_det_J_time_weight.T*c_e_D5
    f2_mechanical = grad_shape_func_y_times_det_J_time_weight.T*c_e_D2+grad_shape_func_x_times_det_J_time_weight.T*c_e_D4+grad_shape_func_z_times_det_J_time_weight.T*c_e_D6
    f3_mechanical = grad_shape_func_z_times_det_J_time_weight.T*c_e_D3+grad_shape_func_x_times_det_J_time_weight.T*c_e_D5+grad_shape_func_y_times_det_J_time_weight.T*c_e_D6
    f_mechanical = np.concatenate((f1_mechanical, f2_mechanical, f3_mechanical))
    # f_mechanical = f_mechanical[reorder_index, :]

    return f_mechanical
