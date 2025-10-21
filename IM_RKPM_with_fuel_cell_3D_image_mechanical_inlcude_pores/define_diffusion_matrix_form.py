import time
start_time = time.time()
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



def diffusion_matrix(shape_func, shape_func_times_det_J_time_weight,grad_shape_func_x,D_R11,grad_shape_func_y,D_R21,grad_shape_func_x_times_det_J_time_weight,D_R12,D_R22,grad_shape_func_y_times_det_J_time_weight,\
                     dD_dc_R11,dD_dc_R21,c_n1,dD_dc_R12,dD_dc_R22,shape_func_b,djbv_deta,dE_eq_dc,shape_func_b_times_det_J_b_time_weight,djbv_dj0,dj0_dc,j_BV,j_applied,x_G_b,k_con,\
                        dt,Fday,c_n,phi_n1):

    M_matrix = shape_func.T*shape_func_times_det_J_time_weight          # sparse matrix
    K_cc = ((grad_shape_func_x).multiply(D_R11)+(grad_shape_func_y).multiply(D_R21)).T*grad_shape_func_x_times_det_J_time_weight+((grad_shape_func_x).multiply(D_R12)+(grad_shape_func_y).multiply(D_R22)).T*grad_shape_func_y_times_det_J_time_weight   # sparse matrix
        
    K_cc_D = (((grad_shape_func_x).multiply(dD_dc_R11)+(grad_shape_func_y).multiply(dD_dc_R21)).multiply(grad_shape_func_x*c_n1)+((grad_shape_func_x).multiply(dD_dc_R12)+(grad_shape_func_y).multiply(dD_dc_R22)).multiply(grad_shape_func_y*c_n1)).T*shape_func_times_det_J_time_weight # sparse matrix
    
    K_cp_Eeq = (shape_func_b.multiply((djbv_deta*dE_eq_dc))).T*shape_func_b_times_det_J_b_time_weight # sparse matrix
    K_cp_j0 = (shape_func_b.multiply((-djbv_dj0*dj0_dc))).T*shape_func_b_times_det_J_b_time_weight# sparse matrix
    K_cp_BV = (shape_func_b.multiply((-djbv_deta))).T*shape_func_b_times_det_J_b_time_weight # sparse matrix
    f_c = shape_func_b_times_det_J_b_time_weight.T*(-j_BV)          # array
    f_phi = shape_func_b_times_det_J_b_time_weight.T*(-(j_BV+j_applied*np.array(np.ones((np.shape(x_G_b)[0],1))))) # array
    K_phiphi = (grad_shape_func_x.T*grad_shape_func_x_times_det_J_time_weight+grad_shape_func_y.T*grad_shape_func_y_times_det_J_time_weight).multiply(k_con) # sparse
    

    K11 = M_matrix + K_cc.multiply(dt)-(K_cp_j0+K_cp_Eeq).multiply(dt/Fday)+K_cc_D.multiply(dt)
    K12 = K_cp_BV.multiply(-dt/Fday)
    K21 = -K_cp_Eeq-K_cp_j0
    K22 = K_phiphi-K_cp_BV

    f1 = f_c*dt/Fday-(K_cc*c_n1)*dt-M_matrix*(c_n1-c_n)
    f2 = f_phi-K_phiphi*phi_n1

    K = bmat([[K11, K12], [K21, K22]]) 
    f = np.concatenate((f1, f2))

    return K,f



def diffusion_matrix_fuel_cell(dimention, point_or_line_source, shape_func_point_or_line_nodes, g_diretchlet, beta_Nitsche, normal_vector_x, normal_vector_y, global_diffusion,grad_shape_func_x,grad_shape_func_y,grad_shape_func_x_times_det_J_time_weight,grad_shape_func_y_times_det_J_time_weight,\
                     shape_func_b,shape_func_b_times_det_J_b_time_weight,grad_shape_func_b_x_times_det_J_b_time_weight, grad_shape_func_b_y_times_det_J_b_time_weight, shape_func_inter_times_det_J_b_time_weight = None, interface_source=None, grad_shape_func_z=None, grad_shape_func_z_times_det_J_time_weight=None, grad_shape_func_b_z_times_det_J_b_time_weight=None, normal_vector_z=None):

    # print('K1')
    K1 = ((grad_shape_func_x_times_det_J_time_weight).multiply(global_diffusion)).T*grad_shape_func_x+((grad_shape_func_y_times_det_J_time_weight).multiply(global_diffusion)).T*grad_shape_func_y
    if dimention == 3:
        K1 += ((grad_shape_func_z_times_det_J_time_weight).multiply(global_diffusion)).T*grad_shape_func_z

    # print(np.shape(normal_vector_x))
    K2 = -((grad_shape_func_b_x_times_det_J_b_time_weight).multiply(normal_vector_x)+grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)).T*shape_func_b
    if dimention == 3:
        K2 -= (grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)).T*shape_func_b

    # print('K3')
    K3 = (shape_func_b.multiply(beta_Nitsche)).T*shape_func_b_times_det_J_b_time_weight

    K = K1+K2+K3

    # print('f1')
    f1 = ((shape_func_b_times_det_J_b_time_weight.multiply(beta_Nitsche)).T)*g_diretchlet
    
    # print('f2')
    f2 = -(grad_shape_func_b_x_times_det_J_b_time_weight.multiply(normal_vector_x)+grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)).T*g_diretchlet
    if dimention == 3:
        f2 -= (grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)).T*g_diretchlet

    # when the point source is expressed in delta function times point source value (body source)
    f3 = shape_func_point_or_line_nodes.T*point_or_line_source

    # interface source (surface source)
    f4 = -shape_func_inter_times_det_J_b_time_weight.T*interface_source


    # print('ff')
    f = f1+f2+f3+f4

    return K,f

def diffusion_matrix_fuel_cell_distributed_point_source(dimention, distributed_point_or_line_source, shape_func_distributed_point_or_line_nodes, g_diretchlet, beta_Nitsche, normal_vector_x, normal_vector_y, global_diffusion,grad_shape_func_x,grad_shape_func_y,grad_shape_func_x_times_det_J_time_weight,grad_shape_func_y_times_det_J_time_weight,\
                     shape_func_b,shape_func_b_times_det_J_b_time_weight,grad_shape_func_b_x_times_det_J_b_time_weight, grad_shape_func_b_y_times_det_J_b_time_weight, shape_func_inter_times_det_J_b_time_weight = None, interface_source=None, grad_shape_func_z=None, grad_shape_func_z_times_det_J_time_weight=None, grad_shape_func_b_z_times_det_J_b_time_weight=None, normal_vector_z=None):

    # print('K1')
    K1 = ((grad_shape_func_x_times_det_J_time_weight).multiply(global_diffusion)).T*grad_shape_func_x+((grad_shape_func_y_times_det_J_time_weight).multiply(global_diffusion)).T*grad_shape_func_y
    if dimention == 3:
        K1 += ((grad_shape_func_z_times_det_J_time_weight).multiply(global_diffusion)).T*grad_shape_func_z

    # print(np.shape(normal_vector_x))
    K2 = -((grad_shape_func_b_x_times_det_J_b_time_weight).multiply(normal_vector_x)+grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)).T*shape_func_b
    if dimention == 3:
        K2 -= (grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)).T*shape_func_b

    # print('K3')
    K3 = (shape_func_b.multiply(beta_Nitsche)).T*shape_func_b_times_det_J_b_time_weight

    K = K1+K2+K3

    # print('f1')
    f1 = ((shape_func_b_times_det_J_b_time_weight.multiply(beta_Nitsche)).T)*g_diretchlet
    
    # print('f2')
    f2 = -(grad_shape_func_b_x_times_det_J_b_time_weight.multiply(normal_vector_x)+grad_shape_func_b_y_times_det_J_b_time_weight.multiply(normal_vector_y)).T*g_diretchlet
    if dimention == 3:
        f2 -= (grad_shape_func_b_z_times_det_J_b_time_weight.multiply(normal_vector_z)).T*g_diretchlet

    # when the point source is expressed in delta function times point source value (surface slource)
    f3 = -shape_func_distributed_point_or_line_nodes.T*distributed_point_or_line_source

    # interface source (surface source)
    f4 = -shape_func_inter_times_det_J_b_time_weight.T*interface_source


    # print('ff')
    f = f1+f2+f3+f4

    return K,f
