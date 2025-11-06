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
# def modify_shape_func_node_removal(itt, x_nodes, x_G, M_modi, M_modi_P_x, M_modi_P_y, phi_scaled, phi_x_scaled, phi_y_scaled, shape_func_row_index_to_be_modified, shape_func_column_index_to_be_modified, HT0, HT1, HT2, differential_method, IM_RKPM, det_J_time_weight, shape_func, shape_func_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight, damaged_interface_nodes_id):

#     ###################################################################################
#     # calculate M matrix at gauss points whose shape func should be modified
#     ###################################################################################

#     # modify the kernal function, the deleted nodes id is N, the N column of kernal func matrix is modified to zero
#     phi_scaled[:, damaged_interface_nodes_id] = 0.0
#     phi_x_scaled[:, damaged_interface_nodes_id] = 0.0
#     phi_y_scaled[:, damaged_interface_nodes_id] = 0.0

#     H_sacling_factor = 1.0e-6

#     for i_m in range(np.shape(shape_func_row_index_to_be_modified)[0]):
#         i = shape_func_row_index_to_be_modified[i_m]
   
#         for j in range(np.shape(x_nodes)[0]):

#             x_I = x_nodes[j]

#             H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
#             H = np.transpose(H_T)

#             HT_P_x = np.array([0,1,0],dtype=np.float64)/H_sacling_factor # partial H partial x
#             HT_P_y = np.array([0,0,1],dtype=np.float64)/H_sacling_factor # partial H partial y

#             H_P_x = np.transpose(HT_P_x)
#             H_P_y = np.transpose(HT_P_y)

#             for ii in range(3):
#                 for jj in range(3):
#                     # print(M_modi[i_m][ii][jj])
#                     # print(H[ii])
#                     # print(H_T[jj])
#                     # print(phi_scaled[i, j])
                    
#                     M_modi[i_m][ii][jj] = M_modi[i_m][ii][jj] + H[ii]*H_T[jj]*phi_scaled[i, j]
#                     M_modi_P_x[i_m][ii][jj] = M_modi_P_x[i_m][ii][jj] + H[ii]*H_T[jj]*phi_x_scaled[i,j] + H_P_x[ii]*H_T[jj]*phi_scaled[i, j] + H[ii]*HT_P_x[jj]*phi_scaled[i, j]
#                     M_modi_P_y[i_m][ii][jj] = M_modi_P_y[i_m][ii][jj] + H[ii]*H_T[jj]*phi_y_scaled[i,j] + H_P_y[ii]*H_T[jj]*phi_scaled[i, j] + H[ii]*HT_P_y[jj]*phi_scaled[i, j]

            
#         # compute the shape function and the gradient of shape function
#         j = shape_func_column_index_to_be_modified[i_m]
#         x_I = x_nodes[j]

#         H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
#         H = np.transpose(H_T)

#         HT_P_x = np.array([0,1,0],dtype=np.float64)/H_sacling_factor # partial H partial x
#         HT_P_y = np.array([0,0,1],dtype=np.float64)/H_sacling_factor # partial H partial y

#         H_P_x = np.transpose(HT_P_x)
#         H_P_y = np.transpose(HT_P_y)
    
#         shape_func_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j]
    
#         if differential_method =='implicite' and IM_RKPM == 'False':
#             grad_shape_func_x_ij = np.dot((np.dot((HT1).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j]
#             grad_shape_func_y_ij = np.dot((np.dot((HT2).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j]

#         else:
#             if differential_method =='direct' or IM_RKPM == 'True':
#                 M_inv_P_x_i = -np.dot(np.dot(np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64), M_modi_P_x[i_m].astype(np.float64)), np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64))
#                 M_inv_P_y_i = -np.dot(np.dot(np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64), M_modi_P_y[i_m].astype(np.float64)), np.linalg.inv(M_modi[i_m].astype(np.float64)).astype(np.float64))
#                 grad_shape_func_x_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_x_scaled[i, j] +\
#                                     np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_x_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j] +\
#                                     np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H_P_x.astype(np.float64))*phi_scaled[i, j]
#                 grad_shape_func_y_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_y_scaled[i, j] +\
#                                     np.dot((np.dot((HT0).astype(np.float64), (M_inv_P_y_i).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_scaled[i, j] +\
#                                     np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M_modi[i_m])).astype(np.float64))).astype(np.float64), H_P_y.astype(np.float64))*phi_scaled[i, j]
#             else:
#                 print('differential method is not defined')

#         shape_func[i,j] = shape_func_ij
#         grad_shape_func_x[i,j] = grad_shape_func_x_ij
#         grad_shape_func_y[i,j] = grad_shape_func_y_ij
#         shape_func_times_det_J_time_weight[i,j] = shape_func_ij*det_J_time_weight[i]
#         grad_shape_func_x_times_det_J_time_weight[i,j] = grad_shape_func_x_ij*det_J_time_weight[i]
#         grad_shape_func_y_times_det_J_time_weight[i,j] = grad_shape_func_y_ij*det_J_time_weight[i]

#     shape_func[:, damaged_interface_nodes_id] = 0.0
#     shape_func_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0
#     grad_shape_func_x[:, damaged_interface_nodes_id] = 0.0
#     grad_shape_func_y[:, damaged_interface_nodes_id] = 0.0
#     grad_shape_func_x_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0
#     grad_shape_func_y_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0

#     return phi_scaled, phi_x_scaled, phi_y_scaled, shape_func, shape_func_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight


# @jit(nopython=True)
def modify_shape_func_node_removal(itt, x_nodes, x_G, M_modi, M_modi_P_x, M_modi_P_y, phi_scaled, phi_x_scaled, phi_y_scaled, shape_func_row_index_to_be_modified, shape_func_column_index_to_be_modified, HT0, HT1, HT2, differential_method, IM_RKPM, det_J_time_weight, shape_func, shape_func_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight, damaged_interface_nodes_id):

    # Convert inputs to numpy arrays for vectorization
    x_nodes = np.array(x_nodes)
    x_G = np.array(x_G)
    phi_scaled = np.array(phi_scaled)
    phi_x_scaled = np.array(phi_x_scaled)
    phi_y_scaled = np.array(phi_y_scaled)
    shape_func_row_index_to_be_modified = np.array(shape_func_row_index_to_be_modified)
    shape_func_column_index_to_be_modified = np.array(shape_func_column_index_to_be_modified)
    damaged_interface_nodes_id = np.array(damaged_interface_nodes_id)
    HT0 = np.array(HT0, dtype=np.float64)
    det_J_time_weight = np.array(det_J_time_weight)
    
    # Zero out damaged interface nodes - VECTORIZED
    phi_scaled[:, damaged_interface_nodes_id] = 0.0
    phi_x_scaled[:, damaged_interface_nodes_id] = 0.0
    phi_y_scaled[:, damaged_interface_nodes_id] = 0.0
    
    H_scaling_factor = 1.0e-6
    n_modified = len(shape_func_row_index_to_be_modified)
    n_nodes = x_nodes.shape[0]
    
    if n_modified > 0:
        # Get modified Gauss point indices and coordinates
        i_modified = shape_func_row_index_to_be_modified  # (n_modified,)
        x_G_modified = x_G[i_modified]  # (n_modified, 2)
        
        # Vectorized H matrix computation for ALL modified point-node combinations - NO LOOPS
        dx_all = x_G_modified[:, None, 0] - x_nodes[None, :, 0]  # (n_modified, n_nodes)
        dy_all = x_G_modified[:, None, 1] - x_nodes[None, :, 1]  # (n_modified, n_nodes)
        
        # Vectorized H matrices for all combinations
        H_T_all = np.zeros((n_modified, n_nodes, 3), dtype=np.float64)
        H_T_all[:, :, 0] = 1.0
        H_T_all[:, :, 1] = dx_all / H_scaling_factor
        H_T_all[:, :, 2] = dy_all / H_scaling_factor
        H_all = H_T_all  # H is transpose of H_T
        
        # H derivative vectors (constant)
        HT_P_x = np.array([0, 1.0/H_scaling_factor, 0], dtype=np.float64)
        HT_P_y = np.array([0, 0, 1.0/H_scaling_factor], dtype=np.float64)
        
        # Get phi values for modified Gauss points - VECTORIZED
        phi_vals_all = phi_scaled[i_modified]  # (n_modified, n_nodes)
        phi_x_vals_all = phi_x_scaled[i_modified]  # (n_modified, n_nodes)
        phi_y_vals_all = phi_y_scaled[i_modified]  # (n_modified, n_nodes)
        
        # COMPLETELY VECTORIZED moment matrix updates - NO LOOPS AT ALL
        # Compute outer products for ALL modified points and ALL nodes simultaneously
        H_outer_HT_all = H_all[:, :, :, None] * H_T_all[:, :, None, :]  # (n_modified, n_nodes, 3, 3)
        
        # Broadcast derivative terms
        HT_P_x_broadcast = HT_P_x[None, None, :, None]  # (1, 1, 3, 1)
        HT_P_y_broadcast = HT_P_y[None, None, :, None]  # (1, 1, 3, 1)
        HT_P_x_outer_HT = HT_P_x_broadcast * H_T_all[:, :, None, :]  # (n_modified, n_nodes, 3, 3)
        H_outer_HT_P_x = H_all[:, :, :, None] * HT_P_x[None, None, None, :]  # (n_modified, n_nodes, 3, 3)
        HT_P_y_outer_HT = HT_P_y_broadcast * H_T_all[:, :, None, :]  # (n_modified, n_nodes, 3, 3)
        H_outer_HT_P_y = H_all[:, :, :, None] * HT_P_y[None, None, None, :]  # (n_modified, n_nodes, 3, 3)
        
        # Scale by phi values and sum over all nodes - COMPLETELY VECTORIZED
        M_contributions = H_outer_HT_all * phi_vals_all[:, :, None, None]
        M_updates = np.sum(M_contributions, axis=1)  # Sum over nodes: (n_modified, 3, 3)
        M_modi += M_updates
        
        M_P_x_contributions = (H_outer_HT_all * phi_x_vals_all[:, :, None, None] +
                              HT_P_x_outer_HT * phi_vals_all[:, :, None, None] +
                              H_outer_HT_P_x * phi_vals_all[:, :, None, None])
        M_P_x_updates = np.sum(M_P_x_contributions, axis=1)  # (n_modified, 3, 3)
        M_modi_P_x += M_P_x_updates
        
        M_P_y_contributions = (H_outer_HT_all * phi_y_vals_all[:, :, None, None] +
                              HT_P_y_outer_HT * phi_vals_all[:, :, None, None] +
                              H_outer_HT_P_y * phi_vals_all[:, :, None, None])
        M_P_y_updates = np.sum(M_P_y_contributions, axis=1)  # (n_modified, 3, 3)
        M_modi_P_y += M_P_y_updates
        
        # Vectorized shape function computation for modified entries - NO LOOPS
        j_modified = shape_func_column_index_to_be_modified  # (n_modified,)
        
        # Get specific H matrices for the modified (i,j) pairs
        dx_specific = x_G_modified[:, 0] - x_nodes[j_modified, 0]  # (n_modified,)
        dy_specific = x_G_modified[:, 1] - x_nodes[j_modified, 1]  # (n_modified,)
        
        H_T_specific = np.zeros((n_modified, 3), dtype=np.float64)
        H_T_specific[:, 0] = 1.0
        H_T_specific[:, 1] = dx_specific / H_scaling_factor
        H_T_specific[:, 2] = dy_specific / H_scaling_factor
        H_specific = H_T_specific  # H is transpose of H_T
        
        # Get phi values for specific modified entries
        phi_specific = phi_scaled[i_modified, j_modified]  # (n_modified,)
        phi_x_specific = phi_x_scaled[i_modified, j_modified]  # (n_modified,)
        phi_y_specific = phi_y_scaled[i_modified, j_modified]  # (n_modified,)
        
        # Vectorized matrix inversions with robust handling - NO LOOPS
        M_inv_modified = np.zeros_like(M_modi, dtype=np.float64)
        
        # Use vectorized operations where possible, fallback to element-wise for robustness
        det_M = np.linalg.det(M_modi.astype(np.float64))
        well_conditioned = np.abs(det_M) > 1e-12
        
        # Vectorized inversion for well-conditioned matrices
        if np.any(well_conditioned):
            M_inv_modified[well_conditioned] = np.linalg.inv(M_modi[well_conditioned].astype(np.float64))
        
        # Pseudo-inverse for ill-conditioned matrices
        if np.any(~well_conditioned):
            ill_conditioned_indices = np.where(~well_conditioned)[0]
            # This part needs element-wise processing due to different matrix conditions
            M_inv_modified[ill_conditioned_indices] = np.array([
                np.linalg.pinv(M_modi[idx].astype(np.float64), rcond=1e-12) 
                for idx in ill_conditioned_indices
            ])
        
        # Vectorized shape function computation
        HT0_M_inv = np.einsum('i,jik->jk', HT0, M_inv_modified)  # (n_modified, 3)
        shape_func_values = np.sum(HT0_M_inv * H_specific, axis=1) * phi_specific
        
        # Vectorized gradient computation
        if differential_method == 'implicite' and IM_RKPM == 'False':
            HT1 = np.array(HT1, dtype=np.float64)
            HT2 = np.array(HT2, dtype=np.float64)
            
            HT1_M_inv = np.einsum('i,jik->jk', HT1, M_inv_modified)  # (n_modified, 3)
            HT2_M_inv = np.einsum('i,jik->jk', HT2, M_inv_modified)  # (n_modified, 3)
            
            grad_shape_func_x_values = np.sum(HT1_M_inv * H_specific, axis=1) * phi_specific
            grad_shape_func_y_values = np.sum(HT2_M_inv * H_specific, axis=1) * phi_specific
            
        else:  # differential_method == 'direct' or IM_RKPM == 'True'
            # Vectorized M_inv derivatives
            M_inv_M_P_x = np.matmul(M_inv_modified, M_modi_P_x)
            M_inv_P_x_modified = -np.matmul(M_inv_M_P_x, M_inv_modified)
            
            M_inv_M_P_y = np.matmul(M_inv_modified, M_modi_P_y)
            M_inv_P_y_modified = -np.matmul(M_inv_M_P_y, M_inv_modified)
            
            # Three terms for gradient computation - VECTORIZED
            term1_x = np.sum(HT0_M_inv * H_specific, axis=1) * phi_x_specific
            term1_y = np.sum(HT0_M_inv * H_specific, axis=1) * phi_y_specific
            
            HT0_M_inv_P_x = np.einsum('i,jik->jk', HT0, M_inv_P_x_modified)  # (n_modified, 3)
            HT0_M_inv_P_y = np.einsum('i,jik->jk', HT0, M_inv_P_y_modified)  # (n_modified, 3)
            term2_x = np.sum(HT0_M_inv_P_x * H_specific, axis=1) * phi_specific
            term2_y = np.sum(HT0_M_inv_P_y * H_specific, axis=1) * phi_specific
            
            HT0_M_inv_H_P_x = np.dot(HT0_M_inv, HT_P_x)  # (n_modified,)
            HT0_M_inv_H_P_y = np.dot(HT0_M_inv, HT_P_y)  # (n_modified,)
            term3_x = HT0_M_inv_H_P_x * phi_specific
            term3_y = HT0_M_inv_H_P_y * phi_specific
            
            grad_shape_func_x_values = term1_x + term2_x + term3_x
            grad_shape_func_y_values = term1_y + term2_y + term3_y
        
        # Update shape functions - VECTORIZED ARRAY INDEXING
        shape_func[i_modified, j_modified] = shape_func_values
        grad_shape_func_x[i_modified, j_modified] = grad_shape_func_x_values
        grad_shape_func_y[i_modified, j_modified] = grad_shape_func_y_values
        
        # Vectorized multiplication by det_J_time_weight
        det_J_specific = det_J_time_weight[i_modified]
        shape_func_times_det_J_time_weight[i_modified, j_modified] = shape_func_values * det_J_specific
        grad_shape_func_x_times_det_J_time_weight[i_modified, j_modified] = grad_shape_func_x_values * det_J_specific
        grad_shape_func_y_times_det_J_time_weight[i_modified, j_modified] = grad_shape_func_y_values * det_J_specific
    
    # Zero out damaged interface nodes in all shape function arrays - VECTORIZED
    shape_func[:, damaged_interface_nodes_id] = 0.0
    shape_func_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_x[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_y[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_x_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0
    grad_shape_func_y_times_det_J_time_weight[:, damaged_interface_nodes_id] = 0.0

    return phi_scaled, phi_x_scaled, phi_y_scaled, shape_func, shape_func_times_det_J_time_weight, grad_shape_func_x, grad_shape_func_y, grad_shape_func_x_times_det_J_time_weight, grad_shape_func_y_times_det_J_time_weight