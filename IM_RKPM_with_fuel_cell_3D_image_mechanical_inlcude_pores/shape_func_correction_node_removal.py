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
def modify_shape_func_node_removal(itt, x_nodes, x_G, M_modi, M_modi_P_x, M_modi_P_y, phi_scaled, phi_x_scaled, phi_y_scaled, shape_func_row_index_to_be_modified, shape_func_column_index_to_be_modified, HT0, HT1, HT2, differential_method, IM_RKPM, det_J_time_weight, shape_func, shape_func_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight, damaged_interface_nodes_id):

    ###################################################################################
    # calculate M matrix at gauss points whose shape func should be modified
    ###################################################################################

    # modify the kernal function, the deleted nodes id is N, the N column of kernal func matrix is modified to zero
    phi_scaled[:, damaged_interface_nodes_id] = 0.0
    phi_x_scaled[:, damaged_interface_nodes_id] = 0.0
    phi_y_scaled[:, damaged_interface_nodes_id] = 0.0

    H_sacling_factor = 1.0e-6

    for i_m in range(np.shape(shape_func_row_index_to_be_modified)[0]):
        i = shape_func_row_index_to_be_modified[i_m]
   
        for j in range(np.shape(x_nodes)[0]):

            x_I = x_nodes[j]

            H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
            H = np.transpose(H_T)

            HT_P_x = np.array([0,1,0],dtype=np.float64)/H_sacling_factor # partial H partial x
            HT_P_y = np.array([0,0,1],dtype=np.float64)/H_sacling_factor # partial H partial y

            H_P_x = np.transpose(HT_P_x)
            H_P_y = np.transpose(HT_P_y)

            for ii in range(3):
                for jj in range(3):
                    # print(M_modi[i_m][ii][jj])
                    # print(H[ii])
                    # print(H_T[jj])
                    # print(phi_scaled[i, j])
                    
                    M_modi[i_m][ii][jj] = M_modi[i_m][ii][jj] + H[ii]*H_T[jj]*phi_scaled[i, j]
                    M_modi_P_x[i_m][ii][jj] = M_modi_P_x[i_m][ii][jj] + H[ii]*H_T[jj]*phi_x_scaled[i,j] + H_P_x[ii]*H_T[jj]*phi_scaled[i, j] + H[ii]*HT_P_x[jj]*phi_scaled[i, j]
                    M_modi_P_y[i_m][ii][jj] = M_modi_P_y[i_m][ii][jj] + H[ii]*H_T[jj]*phi_y_scaled[i,j] + H_P_y[ii]*H_T[jj]*phi_scaled[i, j] + H[ii]*HT_P_y[jj]*phi_scaled[i, j]

            
        # compute the shape function and the gradient of shape function
        j = shape_func_column_index_to_be_modified[i_m]
        x_I = x_nodes[j]

        H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
        H = np.transpose(H_T)

        HT_P_x = np.array([0,1,0],dtype=np.float64)/H_sacling_factor # partial H partial x
        HT_P_y = np.array([0,0,1],dtype=np.float64)/H_sacling_factor # partial H partial y

        H_P_x = np.transpose(HT_P_x)
        H_P_y = np.transpose(HT_P_y)
    
        shape_func_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j]
    
        if differential_method =='implicite' and IM_RKPM == 'False':
            grad_shape_func_x_ij = np.dot((np.dot((HT1).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j]
            grad_shape_func_y_ij = np.dot((np.dot((HT2).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j]

        else:
            if differential_method =='direct' or IM_RKPM == 'True':
                M_inv_P_x_i = -np.dot(np.dot(np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64), M_modi_P_x[i_m].astype(np.float64)), np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64))
                M_inv_P_y_i = -np.dot(np.dot(np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64), M_modi_P_y[i_m].astype(np.float64)), np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64))
                grad_shape_func_x_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_x_scaled[i, j] +\
                                    np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_x_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j] +\
                                    np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H_P_x.astype(np.float64))*phi_scaled[i, j]
                grad_shape_func_y_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_y_scaled[i, j] +\
                                    np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_y_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j] +\
                                    np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H_P_y.astype(np.float64))*phi_scaled[i, j]
            else:
                print('differential method is not defined')

        shape_func[i,j] = shape_func_ij
        grad_shape_func_x[i,j] = grad_shape_func_x_ij
        grad_shape_func_y[i,j] = grad_shape_func_y_ij
        shape_func_times_det_J_time_weight[i,j] = shape_func_ij*det_J_time_weight[i]
        grad_shape_func_x_times_det_J_time_weight[i,j] = grad_shape_func_x_ij*det_J_time_weight[i]
        grad_shape_func_y_times_det_J_time_weight[i,j] = grad_shape_func_y_ij*det_J_time_weight[i]

    shape_func[:, damaged_interface_nodes_id] = 0.0
    shape_func_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_x[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_y[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_x_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_y_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0

    return phi_scaled, phi_x_scaled, phi_y_scaled, shape_func, shape_func_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight


