import time
start_time = time.time()
import numpy as np
from numpy import sign

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig


@jit
def compute_phi_M_int(x_nodes_interface, x_nodes, a, M, M_P_x, M_P_y):
    # x_nodes_interface is the array which save all interface nodes coors, without repeat
    # interface_nodes is the array which save all interface nodes coors, with some repeat

    phi_nonzero_index_row = []
    phi_nonzero_index_column = []
    phi_nonzerovalue_data = []
    phi_P_x_nonzerovalue_data = []
    phi_P_y_nonzerovalue_data = []
    z = []
    z_P_x = []
    z_P_y = []
    phipz = []

    # save_heavyside = []
    # save_heavyside_px = []
    # save_heavyside_py = []

    # saved_dist_func_index = []
    # save_distance_function = []
    # save_distance_function_dx = []
    # save_distance_function_dy = []
    # save_point_D_coor = []

    for i in range(np.shape(x_nodes_interface)[0]):
        
        for j in range(np.shape(x_nodes)[0]):

            z_ij = (((x_nodes_interface[i,0]-x_nodes[j,0])**2+(x_nodes_interface[i,1]-x_nodes[j,1])**2)**0.5)/a[j]
            z_ij_P_x = (x_nodes_interface[i,0]-x_nodes[j,0])/(a[j]*z_ij*a[j]+2.220446049250313e-16)              # partial z partial x, add the small number to force the term with machine accuracy
            z_ij_P_y = (x_nodes_interface[i,1]-x_nodes[j,1])/(a[j]*z_ij*a[j]+2.220446049250313e-16)              # partial z partial y

            x_I = x_nodes[j]

            H_T = np.array([1, (x_nodes_interface[i][0]-x_I[0]), (x_nodes_interface[i][1]-x_I[1])],dtype=np.float64)
            H = np.transpose(H_T)

            HT_P_x = np.array([0,1,0],dtype=np.float64) # partial H partial x
            HT_P_y = np.array([0,0,1],dtype=np.float64) # partial H partial y

            H_P_x = np.transpose(HT_P_x)
            H_P_y = np.transpose(HT_P_y)

            if z_ij >= 0 and z_ij < 0.5:
                
                phi_ij = 2.0/3-4*z_ij**2+4*z_ij**3
                phi_P_z = -8.0*z_ij+12.0*z_ij**2                       # partial phi partial z
            else:
                if z_ij<=1 and z_ij>=0.5:
                    phi_ij = 4.0/3-4*z_ij+4*z_ij**2-4.0/3*z_ij**3
                    phi_P_z = -4+8*z_ij-4*z_ij**2

            if z_ij >= 0 and z_ij <= 1.0:
                
                phi_nonzerovalue_data.append(phi_ij)
                phi_nonzero_index_row.append(i)
                phi_nonzero_index_column.append(j)
                phi_P_x_ij = phi_P_z*z_ij_P_x
                phi_P_y_ij = phi_P_z*z_ij_P_y
                phi_P_x_nonzerovalue_data.append(phi_P_x_ij)    # partial phi partial x
                phi_P_y_nonzerovalue_data.append(phi_P_y_ij)    # partial phi partial y
                z.append(z_ij)
                z_P_x.append(z_ij_P_x)
                z_P_y.append(z_ij_P_y)
                phipz.append(phi_P_z)
                for ii in range(3):
                    for jj in range(3):
                        M[i][ii][jj] = M[i][ii][jj] + H[ii]*H_T[jj]*phi_ij
                        M_P_x[i][ii][jj] = M_P_x[i][ii][jj] + H[ii]*H_T[jj]*phi_P_x_ij + H_P_x[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_x[jj]*phi_ij
                        M_P_y[i][ii][jj] = M_P_y[i][ii][jj] + H[ii]*H_T[jj]*phi_P_y_ij + H_P_y[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_y[jj]*phi_ij
                
    return phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y

    # return save_point_D_coor, save_distance_function,save_distance_function_dx,save_distance_function_dy, phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y


# @jit  # this is taking so long time, we are vectorizing this part
def shape_grad_shape_func_int(x_nodes_interface,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight):
    shape_func_value = []
    shape_func_times_det_J_time_weight_value = []
    grad_shape_func_x_value = []
    grad_shape_func_y_value = []
    grad_shape_func_x_times_det_J_time_weight_value = []
    grad_shape_func_y_times_det_J_time_weight_value = []
    for ii in range(num_non_zero_phi_a):
        i = phi_nonzero_index_row[ii]
        j = phi_nonzero_index_column[ii]
            
        # compute the shape function and the gradient of shape function
        x_I = x_nodes[j]

        H_T = np.array([1, (x_nodes_interface[i][0]-x_I[0]), (x_nodes_interface[i][1]-x_I[1])],dtype=np.float64)
        H = np.transpose(H_T)

        HT_P_x = np.array([0,1,0],dtype=np.float64) # partial H partial x
        HT_P_y = np.array([0,0,1],dtype=np.float64) # partial H partial y

        H_P_x = np.transpose(HT_P_x)
        H_P_y = np.transpose(HT_P_y)
        
        shape_func_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]
        
        if differential_method =='implicite':
            grad_shape_func_x_ij = np.dot((np.dot((HT1).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]
            grad_shape_func_y_ij = np.dot((np.dot((HT2).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]

        else:
            if differential_method =='direct':
                M_inv_P_x_i = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_x[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
                M_inv_P_y_i = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_y[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
                grad_shape_func_x_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_P_x_nonzerovalue_data[ii] +\
                                       np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_x_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii] +\
                                       np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H_P_x.astype(np.float64))*phi_nonzerovalue_data[ii]
                grad_shape_func_y_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_P_y_nonzerovalue_data[ii] +\
                                       np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_y_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii] +\
                                       np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H_P_y.astype(np.float64))*phi_nonzerovalue_data[ii]
            else:
                print('differential method is not defined')
        shape_func_value.append(shape_func_ij)
        grad_shape_func_x_value.append(grad_shape_func_x_ij)
        grad_shape_func_y_value.append(grad_shape_func_y_ij)

        shape_func_times_det_J_time_weight_value.append(shape_func_ij*det_J_time_weight[i])
        grad_shape_func_x_times_det_J_time_weight_value.append(grad_shape_func_x_ij*det_J_time_weight[i])
        grad_shape_func_y_times_det_J_time_weight_value.append(grad_shape_func_y_ij*det_J_time_weight[i])

    return shape_func_value, shape_func_times_det_J_time_weight_value, grad_shape_func_x_value, grad_shape_func_y_value, grad_shape_func_x_times_det_J_time_weight_value, grad_shape_func_y_times_det_J_time_weight_value


