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


# @jit
def compute_phi_M_int(x_nodes_interface, x_nodes, a, M, M_P_x, M_P_y):
#     # x_nodes_interface is the array which save all interface nodes coors, without repeat
#     # interface_nodes is the array which save all interface nodes coors, with some repeat

#     phi_nonzero_index_row = []
#     phi_nonzero_index_column = []
#     phi_nonzerovalue_data = []
#     phi_P_x_nonzerovalue_data = []
#     phi_P_y_nonzerovalue_data = []
#     z = []
#     z_P_x = []
#     z_P_y = []
#     phipz = []

#     # save_heavyside = []
#     # save_heavyside_px = []
#     # save_heavyside_py = []

#     # saved_dist_func_index = []
#     # save_distance_function = []
#     # save_distance_function_dx = []
#     # save_distance_function_dy = []
#     # save_point_D_coor = []

#     for i in range(np.shape(x_nodes_interface)[0]):
        
#         for j in range(np.shape(x_nodes)[0]):

#             z_ij = (((x_nodes_interface[i,0]-x_nodes[j,0])**2+(x_nodes_interface[i,1]-x_nodes[j,1])**2)**0.5)/a[j]
#             z_ij_P_x = (x_nodes_interface[i,0]-x_nodes[j,0])/(a[j]*z_ij*a[j]+2.220446049250313e-16)              # partial z partial x, add the small number to force the term with machine accuracy
#             z_ij_P_y = (x_nodes_interface[i,1]-x_nodes[j,1])/(a[j]*z_ij*a[j]+2.220446049250313e-16)              # partial z partial y

#             x_I = x_nodes[j]

#             H_T = np.array([1, (x_nodes_interface[i][0]-x_I[0]), (x_nodes_interface[i][1]-x_I[1])],dtype=np.float64)
#             H = np.transpose(H_T)

#             HT_P_x = np.array([0,1,0],dtype=np.float64) # partial H partial x
#             HT_P_y = np.array([0,0,1],dtype=np.float64) # partial H partial y

#             H_P_x = np.transpose(HT_P_x)
#             H_P_y = np.transpose(HT_P_y)

#             if z_ij >= 0 and z_ij < 0.5:
                
#                 phi_ij = 2.0/3-4*z_ij**2+4*z_ij**3
#                 phi_P_z = -8.0*z_ij+12.0*z_ij**2                       # partial phi partial z
#             else:
#                 if z_ij<=1 and z_ij>=0.5:
#                     phi_ij = 4.0/3-4*z_ij+4*z_ij**2-4.0/3*z_ij**3
#                     phi_P_z = -4+8*z_ij-4*z_ij**2

#             if z_ij >= 0 and z_ij <= 1.0:
                
#                 phi_nonzerovalue_data.append(phi_ij)
#                 phi_nonzero_index_row.append(i)
#                 phi_nonzero_index_column.append(j)
#                 phi_P_x_ij = phi_P_z*z_ij_P_x
#                 phi_P_y_ij = phi_P_z*z_ij_P_y
#                 phi_P_x_nonzerovalue_data.append(phi_P_x_ij)    # partial phi partial x
#                 phi_P_y_nonzerovalue_data.append(phi_P_y_ij)    # partial phi partial y
#                 z.append(z_ij)
#                 z_P_x.append(z_ij_P_x)
#                 z_P_y.append(z_ij_P_y)
#                 phipz.append(phi_P_z)
#                 for ii in range(3):
#                     for jj in range(3):
#                         M[i][ii][jj] = M[i][ii][jj] + H[ii]*H_T[jj]*phi_ij
#                         M_P_x[i][ii][jj] = M_P_x[i][ii][jj] + H[ii]*H_T[jj]*phi_P_x_ij + H_P_x[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_x[jj]*phi_ij
#                         M_P_y[i][ii][jj] = M_P_y[i][ii][jj] + H[ii]*H_T[jj]*phi_P_y_ij + H_P_y[ii]*H_T[jj]*phi_ij + H[ii]*HT_P_y[jj]*phi_ij
                
#     return phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y

#     # return save_point_D_coor, save_distance_function,save_distance_function_dx,save_distance_function_dy, phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y



    n_interface = x_nodes_interface.shape[0]
    n_nodes = x_nodes.shape[0]
    
    # Create all combinations using broadcasting - NO LOOPS
    i_idx = np.arange(n_interface)[:, np.newaxis]  # (n_interface, 1)
    j_idx = np.arange(n_nodes)[np.newaxis, :]      # (1, n_nodes)
    
    # Vectorized distance computation for ALL combinations at once
    dx = x_nodes_interface[i_idx, 0] - x_nodes[j_idx, 0]  # (n_interface, n_nodes)
    dy = x_nodes_interface[i_idx, 1] - x_nodes[j_idx, 1]  # (n_interface, n_nodes)
    dist = np.sqrt(dx**2 + dy**2)
    z_ij = dist / a[j_idx]
    
    # Vectorized z derivatives for ALL combinations
    eps = 2.220446049250313e-16
    z_ij_safe_denom = a[j_idx] * z_ij * a[j_idx] + eps
    z_ij_P_x = dx / z_ij_safe_denom
    z_ij_P_y = dy / z_ij_safe_denom
    
    # Vectorized shape function computation for ALL combinations
    mask1 = (z_ij >= 0) & (z_ij < 0.5)
    mask2 = (z_ij >= 0.5) & (z_ij <= 1.0)
    mask_valid = (z_ij >= 0) & (z_ij <= 1.0)
    
    # Apply shape functions vectorized - NO LOOPS
    phi_ij = np.where(mask1, 
                      2.0/3 - 4*z_ij**2 + 4*z_ij**3,
                      np.where(mask2, 
                               4.0/3 - 4*z_ij + 4*z_ij**2 - 4.0/3*z_ij**3, 
                               0.0))
    
    phi_P_z = np.where(mask1,
                       -8.0*z_ij + 12.0*z_ij**2,
                       np.where(mask2,
                                -4 + 8*z_ij - 4*z_ij**2,
                                0.0))
    
    # Vectorized moment matrix updates using broadcasting - NO LOOPS AT ALL
    # H matrices for ALL interface-node combinations
    H_all = np.zeros((n_interface, n_nodes, 3), dtype=np.float64)
    H_all[:, :, 0] = 1.0
    H_all[:, :, 1] = dx
    H_all[:, :, 2] = dy
    
    # Vectorized outer products for ALL combinations
    H_outer_H = H_all[:, :, :, np.newaxis] * H_all[:, :, np.newaxis, :]  # (n_interface, n_nodes, 3, 3)
    
    # Derivative terms
    HT_P_x = np.array([0.0, 1.0, 0.0])
    HT_P_y = np.array([0.0, 0.0, 1.0])
    
    # Broadcast derivative outer products
    HT_P_x_outer_H = HT_P_x[np.newaxis, np.newaxis, :, np.newaxis] * H_all[:, :, np.newaxis, :]  # (n_interface, n_nodes, 3, 3)
    H_outer_HT_P_x = H_all[:, :, :, np.newaxis] * HT_P_x[np.newaxis, np.newaxis, np.newaxis, :]  # (n_interface, n_nodes, 3, 3)
    HT_P_y_outer_H = HT_P_y[np.newaxis, np.newaxis, :, np.newaxis] * H_all[:, :, np.newaxis, :]  # (n_interface, n_nodes, 3, 3)
    H_outer_HT_P_y = H_all[:, :, :, np.newaxis] * HT_P_y[np.newaxis, np.newaxis, np.newaxis, :]  # (n_interface, n_nodes, 3, 3)
    
    # Compute phi derivatives
    phi_P_x_ij = phi_P_z * z_ij_P_x
    phi_P_y_ij = phi_P_z * z_ij_P_y
    
    # Apply mask and sum contributions - COMPLETELY VECTORIZED
    phi_ij_masked = np.where(mask_valid, phi_ij, 0.0)
    phi_P_x_ij_masked = np.where(mask_valid, phi_P_x_ij, 0.0)
    phi_P_y_ij_masked = np.where(mask_valid, phi_P_y_ij, 0.0)
    
    # Update moment matrices using vectorized summation - NO LOOPS
    M_contributions = H_outer_H * phi_ij_masked[:, :, np.newaxis, np.newaxis]
    M += np.sum(M_contributions, axis=1)  # Sum over nodes for each interface point
    
    M_P_x_contributions = (H_outer_H * phi_P_x_ij_masked[:, :, np.newaxis, np.newaxis] + 
                          HT_P_x_outer_H * phi_ij_masked[:, :, np.newaxis, np.newaxis] + 
                          H_outer_HT_P_x * phi_ij_masked[:, :, np.newaxis, np.newaxis])
    M_P_x += np.sum(M_P_x_contributions, axis=1)
    
    M_P_y_contributions = (H_outer_H * phi_P_y_ij_masked[:, :, np.newaxis, np.newaxis] + 
                          HT_P_y_outer_H * phi_ij_masked[:, :, np.newaxis, np.newaxis] + 
                          H_outer_HT_P_y * phi_ij_masked[:, :, np.newaxis, np.newaxis])
    M_P_y += np.sum(M_P_y_contributions, axis=1)
    
    # Extract valid entries for return values - NO LOOPS
    i_indices, j_indices = np.where(mask_valid)
    phi_nonzero_index_row = i_indices
    phi_nonzero_index_column = j_indices
    phi_nonzerovalue_data = phi_ij[mask_valid]
    phi_P_x_nonzerovalue_data = phi_P_x_ij[mask_valid]
    phi_P_y_nonzerovalue_data = phi_P_y_ij[mask_valid]
    
    return (phi_nonzero_index_row.tolist(), phi_nonzero_index_column.tolist(),
            phi_nonzerovalue_data.tolist(), phi_P_x_nonzerovalue_data.tolist(),
            phi_P_y_nonzerovalue_data.tolist(), M, M_P_x, M_P_y)

# @jit  # this is taking so long time, we are vectorizing this part
def shape_grad_shape_func_int(x_nodes_interface,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight):
    # shape_func_value = []
    # shape_func_times_det_J_time_weight_value = []
    # grad_shape_func_x_value = []
    # grad_shape_func_y_value = []
    # grad_shape_func_x_times_det_J_time_weight_value = []
    # grad_shape_func_y_times_det_J_time_weight_value = []
    # for ii in range(num_non_zero_phi_a):
    #     i = phi_nonzero_index_row[ii]
    #     j = phi_nonzero_index_column[ii]
            
    #     # compute the shape function and the gradient of shape function
    #     x_I = x_nodes[j]

    #     H_T = np.array([1, (x_nodes_interface[i][0]-x_I[0]), (x_nodes_interface[i][1]-x_I[1])],dtype=np.float64)
    #     H = np.transpose(H_T)

    #     HT_P_x = np.array([0,1,0],dtype=np.float64) # partial H partial x
    #     HT_P_y = np.array([0,0,1],dtype=np.float64) # partial H partial y

    #     H_P_x = np.transpose(HT_P_x)
    #     H_P_y = np.transpose(HT_P_y)
        
    #     shape_func_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]
        
    #     if differential_method =='implicite':
    #         grad_shape_func_x_ij = np.dot((np.dot((HT1).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]
    #         grad_shape_func_y_ij = np.dot((np.dot((HT2).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]

    #     else:
    #         if differential_method =='direct':
    #             M_inv_P_x_i = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_x[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
    #             M_inv_P_y_i = -np.dot(np.dot(np.linalg.inv(M[i].astype(np.float64)).astype(np.float64), M_P_y[i].astype(np.float64)), np.linalg.inv(M[i].astype(np.float64)).astype(np.float64))
    #             grad_shape_func_x_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_P_x_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_x_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H_P_x.astype(np.float64))*phi_nonzerovalue_data[ii]
    #             grad_shape_func_y_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_P_y_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_y_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii] +\
    #                                    np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H_P_y.astype(np.float64))*phi_nonzerovalue_data[ii]
    #         else:
    #             print('differential method is not defined')
    #     shape_func_value.append(shape_func_ij)
    #     grad_shape_func_x_value.append(grad_shape_func_x_ij)
    #     grad_shape_func_y_value.append(grad_shape_func_y_ij)

    #     shape_func_times_det_J_time_weight_value.append(shape_func_ij*det_J_time_weight[i])
    #     grad_shape_func_x_times_det_J_time_weight_value.append(grad_shape_func_x_ij*det_J_time_weight[i])
    #     grad_shape_func_y_times_det_J_time_weight_value.append(grad_shape_func_y_ij*det_J_time_weight[i])

    # return shape_func_value, shape_func_times_det_J_time_weight_value, grad_shape_func_x_value, grad_shape_func_y_value, grad_shape_func_x_times_det_J_time_weight_value, grad_shape_func_y_times_det_J_time_weight_value
    
    
    # def shape_grad_shape_func_int(x_nodes_interface, x_nodes, num_non_zero_phi_a, HT0, M, M_P_x, M_P_y, 
    #                          differential_method, HT1, HT2, phi_nonzerovalue_data, 
    #                          phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, 
    #                          phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight):
    
    # Convert inputs to numpy arrays for vectorization
    phi_nonzero_index_row = np.array(phi_nonzero_index_row)
    phi_nonzero_index_column = np.array(phi_nonzero_index_column)
    phi_nonzerovalue_data = np.array(phi_nonzerovalue_data)
    phi_P_x_nonzerovalue_data = np.array(phi_P_x_nonzerovalue_data)
    phi_P_y_nonzerovalue_data = np.array(phi_P_y_nonzerovalue_data)
    det_J_time_weight = np.array(det_J_time_weight)
    HT0 = np.array(HT0, dtype=np.float64)
    x_nodes_interface = np.array(x_nodes_interface)
    x_nodes = np.array(x_nodes)
    
    # Get indices for vectorized operations
    i_indices = phi_nonzero_index_row  # Interface node indices
    j_indices = phi_nonzero_index_column  # Regular node indices
    n_entries = len(i_indices)
    
    # Vectorized H matrix computation for ALL entries at once
    x_I = x_nodes[j_indices]  # (n_entries, 2)
    x_interface_selected = x_nodes_interface[i_indices]  # (n_entries, 2)
    
    # Compute H_T matrices for all entries at once
    H_T_all = np.zeros((n_entries, 3), dtype=np.float64)
    H_T_all[:, 0] = 1.0
    H_T_all[:, 1] = x_interface_selected[:, 0] - x_I[:, 0]
    H_T_all[:, 2] = x_interface_selected[:, 1] - x_I[:, 1]
    H_all = H_T_all  # H is transpose of H_T
    
    # H derivative vectors (constant for all entries)
    HT_P_x = np.array([0, 1, 0], dtype=np.float64)
    HT_P_y = np.array([0, 0, 1], dtype=np.float64)
    H_P_x = HT_P_x
    H_P_y = HT_P_y
    
    # Get M matrices for relevant interface points and compute inverses
    M_selected = M[i_indices].astype(np.float64)  # (n_entries, 3, 3)
    M_inv_selected = np.linalg.inv(M_selected)  # (n_entries, 3, 3)
    
    # Vectorized shape function computation
    # Compute HT0 @ M_inv for all entries using broadcasting
    HT0_M_inv = np.tensordot(HT0, M_inv_selected, axes=([0], [1]))  # (n_entries, 3)
    # Compute shape function values: (HT0 @ M_inv @ H) * phi for all entries
    shape_func_values = np.sum(HT0_M_inv * H_all, axis=1) * phi_nonzerovalue_data
    
    # Vectorized gradient computation based on differential method
    if differential_method == 'implicite':
        # Use implicit differentiation with HT1, HT2
        HT1 = np.array(HT1, dtype=np.float64)
        HT2 = np.array(HT2, dtype=np.float64)
        
        HT1_M_inv = np.tensordot(HT1, M_inv_selected, axes=([0], [1]))  # (n_entries, 3)
        HT2_M_inv = np.tensordot(HT2, M_inv_selected, axes=([0], [1]))  # (n_entries, 3)
        
        grad_shape_func_x_values = np.sum(HT1_M_inv * H_all, axis=1) * phi_nonzerovalue_data
        grad_shape_func_y_values = np.sum(HT2_M_inv * H_all, axis=1) * phi_nonzerovalue_data
        
    elif differential_method == 'direct':
        # Use direct differentiation with M_P_x, M_P_y
        M_P_x_selected = M_P_x[i_indices].astype(np.float64)  # (n_entries, 3, 3)
        M_P_y_selected = M_P_y[i_indices].astype(np.float64)  # (n_entries, 3, 3)
        
        # Vectorized computation of M_inv derivatives: M_inv_P = -M_inv @ M_P @ M_inv
        M_inv_M_P_x = np.matmul(M_inv_selected, M_P_x_selected)
        M_inv_P_x_selected = -np.matmul(M_inv_M_P_x, M_inv_selected)
        
        M_inv_M_P_y = np.matmul(M_inv_selected, M_P_y_selected)
        M_inv_P_y_selected = -np.matmul(M_inv_M_P_y, M_inv_selected)
        
        # Three terms for gradient computation (vectorized)
        # Term 1: (HT0 @ M_inv @ H) * phi_P
        term1_x = np.sum(HT0_M_inv * H_all, axis=1) * phi_P_x_nonzerovalue_data
        term1_y = np.sum(HT0_M_inv * H_all, axis=1) * phi_P_y_nonzerovalue_data
        
        # Term 2: (HT0 @ M_inv_P @ H) * phi
        HT0_M_inv_P_x = np.tensordot(HT0, M_inv_P_x_selected, axes=([0], [1]))  # (n_entries, 3)
        HT0_M_inv_P_y = np.tensordot(HT0, M_inv_P_y_selected, axes=([0], [1]))  # (n_entries, 3)
        term2_x = np.sum(HT0_M_inv_P_x * H_all, axis=1) * phi_nonzerovalue_data
        term2_y = np.sum(HT0_M_inv_P_y * H_all, axis=1) * phi_nonzerovalue_data
        
        # Term 3: (HT0 @ M_inv @ H_P) * phi
        # H_P_x and H_P_y are constant vectors, so we can broadcast
        HT0_M_inv_H_P_x = np.dot(HT0_M_inv, H_P_x)  # (n_entries,)
        HT0_M_inv_H_P_y = np.dot(HT0_M_inv, H_P_y)  # (n_entries,)
        term3_x = HT0_M_inv_H_P_x * phi_nonzerovalue_data
        term3_y = HT0_M_inv_H_P_y * phi_nonzerovalue_data
        
        # Combine all terms
        grad_shape_func_x_values = term1_x + term2_x + term3_x
        grad_shape_func_y_values = term1_y + term2_y + term3_y
        
    else:
        print('differential method is not defined')
        grad_shape_func_x_values = np.zeros(n_entries)
        grad_shape_func_y_values = np.zeros(n_entries)
    
    # Vectorized multiplication by det_J_time_weight
    det_J_selected = det_J_time_weight[i_indices]
    shape_func_times_det_J_time_weight_values = shape_func_values * det_J_selected
    grad_shape_func_x_times_det_J_time_weight_values = grad_shape_func_x_values * det_J_selected
    grad_shape_func_y_times_det_J_time_weight_values = grad_shape_func_y_values * det_J_selected
    
    # Convert to lists to match original interface
    return (
        shape_func_values.tolist(),
        shape_func_times_det_J_time_weight_values.tolist(),
        grad_shape_func_x_values.tolist(),
        grad_shape_func_y_values.tolist(),
        grad_shape_func_x_times_det_J_time_weight_values.tolist(),
        grad_shape_func_y_times_det_J_time_weight_values.tolist()
    )