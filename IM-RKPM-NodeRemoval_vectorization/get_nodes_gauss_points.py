import time
start_time = time.time()
import numpy as np
from numpy import sign

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit, njit

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

from collections import Counter


# @jit(nopython=True)
def get_x_nodes_single_grain(n_nodes,x_min,x_max,n_intervals,y_min,y_max):
    # x_nodes = []
    # for j in range(n_nodes):
    #     for i in range(n_nodes):
    #         x_nodes.append([x_min+(x_max-x_min)/n_intervals*i, y_min+(y_max-y_min)/n_intervals*j])
    

    # Create 1D arrays for x and y coordinates
    x_coords = np.linspace(x_min, x_max, n_nodes)
    y_coords = np.linspace(y_min, y_max, n_nodes)
    
    # Create meshgrid to get all combinations
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Flatten and combine into coordinate pairs
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Create output array
    x_nodes = np.empty((len(x_flat), 2), dtype=np.float64)
    x_nodes[:, 0] = x_flat
    x_nodes[:, 1] = y_flat

    return x_nodes

@jit
def get_x_nodes_multi_grain(x_min,x_max,y_min,y_max, num_pixels_x, num_pixels_y, img_):
    # define initial RPK nodes
    
    # x_nodes_ini = []
    # for j in range(num_pixels_x):
    #     for i in range(num_pixels_y):
    #         x_nodes_ini.append([x_min+(x_max-x_min)/(num_pixels_x-1)*j, y_min+(y_max-y_min)/(num_pixels_y-1)*i])
    
    # # Create 1D arrays for x and y coordinates
    # x_coords = np.linspace(x_min, x_max, num_pixels_x)
    # y_coords = np.linspace(y_min, y_max, num_pixels_y)
    
    # # Create meshgrid to get all combinations
    # X, Y = np.meshgrid(x_coords, y_coords)
    
    # # Flatten and combine into coordinate pairs
    # x_flat = X.flatten()
    # y_flat = Y.flatten()
    
    # # Create output array
    # x_nodes_ini = np.empty((len(x_flat), 2), dtype=np.float64)
    # x_nodes_ini[:, 0] = x_flat
    # x_nodes_ini[:, 1] = y_flat

    # go through all cells, partition cells if needed
    num_rec_cell = 0
    num_tri_cell = 0

    cell_nodes_list = []    # cell_nodes_list[i] is all nodes coordinates of cell i, 
    grain_id = []           # grain id of each cell, 
    cell_shape = []          # shape of each cell, triangle ('tri') or rectangle ('rec'), 


    bottom_boundary_cell_nodes_list = []  # corresponding to bottom, right, top, left boundaries, 
    right_boundary_cell_nodes_list = [] 
    top_boundary_cell_nodes_list = []
    left_boundary_cell_nodes_list = [] 

    grain_id_left = []
    grain_id_right = []
    grain_id_top = []
    grain_id_bottom = []

    x_nodes_added = []
    x_nodes_added_id = []

    x_nodes = []

    nodes_grain_id = []

    repeated_vertex = []    # when do the gauss integral, the triangle element is treated as rectangle. One of the vertex of triangle was repeated (the first , or the third verex)

    interface_segments = []  # all interface segments

    # go through all nodes
  
    for j in range(num_pixels_y-1):
        for i in range(num_pixels_x-1):

            if [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j] not in x_nodes:
                x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j])
                nodes_grain_id.append(img_[i,j])
            if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j] not in x_nodes:
                x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j])
                nodes_grain_id.append(img_[i+1,j])
            if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)] not in x_nodes:
                x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                nodes_grain_id.append(img_[i+1,j+1])
            if [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)] not in x_nodes:
                x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                nodes_grain_id.append(img_[i,j+1])


            added_nodes_number = 0 # number of added nodes for each cell

            add_node_bottom = 'False'
            add_node_right = 'False'
            add_node_top = 'False'
            add_node_left = 'False'

            if img_[i, j] != img_[i+1, j]:
                added_nodes_number = added_nodes_number+1
                add_node_bottom = 'True'
                
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j] not in x_nodes:
                    x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j])
                    nodes_grain_id.append(img_[i+1, j]) # the gain id of nodes on interface does not matter
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j] not in x_nodes_added:
                    x_nodes_added.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j])
                    x_nodes_added_id.append(len(x_nodes)-1)
            if j == 0:
                if add_node_bottom == 'True':
                    bottom_boundary_cell_nodes_list.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i), x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5)])
                    bottom_boundary_cell_nodes_list.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1)])
                    grain_id_bottom.append(img_[i, j])
                    grain_id_bottom.append(img_[i+1, j])
                else:
                    bottom_boundary_cell_nodes_list.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i), x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1)])
                    grain_id_bottom.append(img_[i, j])

            if img_[i+1, j] != img_[i+1, j+1]:
                added_nodes_number = added_nodes_number+1
                add_node_right = 'True'
                
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)] not in x_nodes:
                    x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)])
                    nodes_grain_id.append(img_[i+1, j])
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)] not in x_nodes_added:
                    x_nodes_added.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)])
                    x_nodes_added_id.append(len(x_nodes)-1)
            if i == num_pixels_x-2:
                if add_node_right == 'True':
                    right_boundary_cell_nodes_list.append([y_min+(y_max-y_min)/(num_pixels_y-1)*(j), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)])
                    right_boundary_cell_nodes_list.append([y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                    grain_id_right.append(img_[i+1, j])
                    grain_id_right.append(img_[i+1, j+1])
                else:
                    right_boundary_cell_nodes_list.append([y_min+(y_max-y_min)/(num_pixels_y-1)*(j), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                    grain_id_right.append(img_[i+1, j])


            if img_[i+1, j+1] != img_[i, j+1]:
                added_nodes_number = added_nodes_number+1
                add_node_top = 'True'
                

                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)] not in x_nodes:
                    x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                    nodes_grain_id.append(img_[i+1, j])
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)] not in x_nodes_added:
                    x_nodes_added.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                    x_nodes_added_id.append(len(x_nodes)-1)

            if j == num_pixels_y-2:
                if add_node_top == 'True':
                    top_boundary_cell_nodes_list.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i), x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5)])
                    top_boundary_cell_nodes_list.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1)])
                    grain_id_top.append(img_[i, j+1])
                    grain_id_top.append(img_[i+1, j+1])
                else:
                    top_boundary_cell_nodes_list.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i), x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1)])
                    grain_id_top.append(img_[i+1, j+1])

            if img_[i, j] != img_[i, j+1]:
                added_nodes_number = added_nodes_number+1
                add_node_left = 'True'
                
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)] not in x_nodes:
                    x_nodes.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)])
                    nodes_grain_id.append(img_[i+1, j])
                if [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)] not in x_nodes_added:
                    x_nodes_added.append([x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)])
                    x_nodes_added_id.append(len(x_nodes)-1)

            if i == 0:
                if add_node_left == 'True':
                    left_boundary_cell_nodes_list.append([y_min+(y_max-y_min)/(num_pixels_y-1)*(j), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)])
                    left_boundary_cell_nodes_list.append([y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                    grain_id_left.append(img_[i, j])
                    grain_id_left.append(img_[i, j+1])
                else:
                    left_boundary_cell_nodes_list.append([y_min+(y_max-y_min)/(num_pixels_y-1)*(j), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)])
                    grain_id_left.append(img_[i, j])

            # if no node should be added
            if added_nodes_number==0: # or added_nodes_number==1:
                cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
               
                grain_id.append(img_[i,j])
                
                cell_shape.append('rec')
                repeated_vertex.append('No')
                num_rec_cell  = num_rec_cell + 1
                    

            if added_nodes_number==2: # interface of two different grains

               
                if (add_node_bottom == 'True' and add_node_top == 'True') or (add_node_left == 'True' and add_node_right == 'True'):
                    # split into four squares
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1
                    
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    if add_node_bottom == 'True' and add_node_top == 'True':
                        interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                       
                    
                            
                        
                    if add_node_left == 'True' and add_node_right == 'True':
                        interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                        
                    

                
                # split into 3 rectangle cells two triangle cells
                if add_node_bottom == 'True' and add_node_right == 'True':
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('tri')
                    
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])

                if add_node_bottom == 'True' and add_node_left == 'True':

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1
                    
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])

                if add_node_top == 'True' and add_node_left == 'True':

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])

                if add_node_top == 'True' and add_node_right == 'True':

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                            [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])

            if added_nodes_number==3: # interface of 3 different grains

                   
                if add_node_left == 'False':
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    
                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])

                if add_node_bottom == 'False':
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])

                if add_node_right == 'False':
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                    [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])

                if add_node_top == 'False':
                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('rec')
                    repeated_vertex.append('No')
                    num_rec_cell  = num_rec_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j+1])
                    cell_shape.append('tri')
                    repeated_vertex.append('first')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                    grain_id.append(img_[i+1,j])
                    cell_shape.append('tri')
                    repeated_vertex.append('three')
                    num_tri_cell  = num_tri_cell + 1

                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)]])
                    interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])

            if added_nodes_number==4: # interface of 3 or 4 different grains

                cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                grain_id.append(img_[i,j])
                cell_shape.append('rec')
                repeated_vertex.append('No')
                num_rec_cell  = num_rec_cell + 1

                cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                grain_id.append(img_[i+1,j])
                cell_shape.append('rec')
                repeated_vertex.append('No')
                num_rec_cell  = num_rec_cell + 1

                cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                grain_id.append(img_[i+1,j+1])
                cell_shape.append('rec')
                repeated_vertex.append('No')
                num_rec_cell  = num_rec_cell + 1

                cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                                                [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                grain_id.append(img_[i,j+1])
                cell_shape.append('rec')
                repeated_vertex.append('No')
                num_rec_cell  = num_rec_cell + 1

                interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                # interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                #                         [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                                        [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])
                # interface_segments.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)],\
                #                         [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]])

    x_nodes = np.array(x_nodes)
    nodes_grain_id = np.array(nodes_grain_id)

    # interface_segments
    x_nodes_added = np.array(x_nodes_added)

    return cell_nodes_list, grain_id, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, cell_shape, num_rec_cell, num_tri_cell, x_nodes, nodes_grain_id, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, repeated_vertex, interface_segments, x_nodes_added, x_nodes_added_id


# get all gauss points in domain 
def x_G_and_def_J_time_weight_structured(n_intervals, x_min,x_max,y_min,y_max,x_G_domain,weight_G_domain):
    # x_G = []      # xy coordinates of gauss points in domain   
    # det_J_time_weight = []    # determin of jacobian
    # for n in range(n_intervals):
    #     for m in range(n_intervals):
    #         # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
    #         x_ver_mn = np.array([x_min+m*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+m*(x_max-x_min)/n_intervals],dtype=np.float64)
    #         y_ver_mn = np.array([y_min+n*(y_max-y_min)/n_intervals, y_min+n*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals],dtype=np.float64)
    #         # calculate the cy coordinates of gauss points in current integration domain
    #         for k in range(len(x_G_domain)):
                
    #             x_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain[k][0])*(1-x_G_domain[k][1]), (1+x_G_domain[k][0])*(1-x_G_domain[k][1]), \
    #                                     (1+x_G_domain[k][0])*(1+x_G_domain[k][1]), (1-x_G_domain[k][0])*(1+x_G_domain[k][1])],dtype=np.float64), np.transpose(x_ver_mn))
    #             y_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain[k][0])*(1-x_G_domain[k][1]), (1+x_G_domain[k][0])*(1-x_G_domain[k][1]), \
    #                                     (1+x_G_domain[k][0])*(1+x_G_domain[k][1]), (1-x_G_domain[k][0])*(1+x_G_domain[k][1])],dtype=np.float64), np.transpose(y_ver_mn))
    #             x_G.append([x_G_mn_k, y_G_mn_k])
    #             J1 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][1]), (1-x_G_domain[k][1]), (1+x_G_domain[k][1]), (-1-x_G_domain[k][1])]), np.transpose(x_ver_mn))
    #             J2 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][1]), (1-x_G_domain[k][1]), (1+x_G_domain[k][1]), (-1-x_G_domain[k][1])]), np.transpose(y_ver_mn))
    #             J3 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][0]), (-1-x_G_domain[k][0]), (1+x_G_domain[k][0]), (1-x_G_domain[k][0])]), np.transpose(x_ver_mn))
    #             J4 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][0]), (-1-x_G_domain[k][0]), (1+x_G_domain[k][0]), (1-x_G_domain[k][0])]), np.transpose(y_ver_mn))

    #             det_J_time_weight.append(np.linalg.det(np.array([[J1, J2],[J3,J4]]))*weight_G_domain[k])
    # --- Setup ---
    x_G_domain = np.array(x_G_domain, dtype=np.float64)           # (num_gauss_points, 2)
    weight_G_domain = np.array(weight_G_domain, dtype=np.float64) # (num_gauss_points,)

    # --- Compute all cell bottom-left indices (n,m) ---
    n_idx = np.arange(n_intervals)
    m_idx = np.arange(n_intervals)
    # Create all (n, m) pairs
    n_idx_grid, m_idx_grid = np.meshgrid(n_idx, m_idx, indexing="ij")
    # Flatten to 1D (to build (n_cells,))
    n_cells = n_intervals * n_intervals
    n_idx_flat = n_idx_grid.ravel()  # (n_cells,)
    m_idx_flat = m_idx_grid.ravel()

    # --- Compute cell corner coordinates (for all cells) ---
    # x vertices: (n_cells, 4)
    h_x = (x_max - x_min) / n_intervals
    x_ver_mn = np.stack([
        x_min + m_idx_flat * h_x,
        x_min + (m_idx_flat + 1) * h_x,
        x_min + (m_idx_flat + 1) * h_x,
        x_min + m_idx_flat * h_x
    ], axis=1)
    # y vertices: (n_cells, 4)
    h_y = (y_max - y_min) / n_intervals
    y_ver_mn = np.stack([
        y_min + n_idx_flat * h_y,
        y_min + n_idx_flat * h_y,
        y_min + (n_idx_flat + 1) * h_y,
        y_min + (n_idx_flat + 1) * h_y
    ], axis=1)

    # --- Compute shape function values (for all Gauss points) ---
    # shape_fn: (num_gauss, 4)
    xi = x_G_domain[:, 0]
    eta = x_G_domain[:, 1]

    shape_fn = np.stack([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ], axis=1) * 0.25     # (num_gauss, 4)

    # Derivatives wrt xi and eta:
    dN_dxi = np.stack([
        -(1 - eta), (1 - eta), (1 + eta), -(1 + eta)
    ], axis=1) * 0.25     # (num_gauss, 4)
    dN_deta = np.stack([
        -(1 - xi), -(1 + xi), (1 + xi), (1 - xi)
    ], axis=1) * 0.25     # (num_gauss, 4)

    # --- Vectorized Gauss point positions for all cells and all points ---
    # Use broadcasting:
    # shape_fn: (num_gauss, 4); x_ver_mn: (n_cells, 4)
    # For all combinations, outer sum: (n_cells, num_gauss, 4)
    # Compute dot product along vertices axis
    #    x_G_mn_k = sum_j shape_fn[k, j] * x_ver_mn[i, j]
    x_Gx = np.tensordot(shape_fn, x_ver_mn, axes=([1], [1])).T    # (n_cells, num_gauss)
    x_Gy = np.tensordot(shape_fn, y_ver_mn, axes=([1], [1])).T    # (n_cells, num_gauss)
    # Flatten: (n_cells * num_gauss,)
    x_G_flat = np.stack([x_Gx.ravel(), x_Gy.ravel()], axis=1)      # (n_cells*num_gauss, 2)

    # --- Jacobian components ---
    J1 = np.tensordot(dN_dxi, x_ver_mn, axes=([1],[1])).T        # (n_cells, num_gauss)
    J2 = np.tensordot(dN_dxi, y_ver_mn, axes=([1],[1])).T
    J3 = np.tensordot(dN_deta, x_ver_mn, axes=([1],[1])).T
    J4 = np.tensordot(dN_deta, y_ver_mn, axes=([1],[1])).T

    # Stack each cell/gauss into (2,2) matrix, flatten to (n_cells*num_gauss, 2, 2)
    # Each entry in result array: [[J1[i,k], J2[i,k]], [J3[i,k], J4[i,k]]]
    J_matrix = np.stack([
        np.stack([J1, J2], axis=2),
        np.stack([J3, J4], axis=2)
    ], axis=2) # shape: (n_cells, num_gauss, 2, 2)
    J_matrix_flat = J_matrix.reshape(-1,2,2)  # (n_cells*num_gauss, 2, 2)
    # Determinants
    det_J = np.linalg.det(J_matrix_flat)     # (n_cells*num_gauss,)
    # weights repeated for each cell
    weight_G_full = np.tile(weight_G_domain, n_cells) # (n_cells*num_gauss,)
    det_J_time_weight = det_J * weight_G_full          # (n_cells*num_gauss,)

    return x_G_flat, det_J_time_weight

# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian on boundaries

# @jit
def x_G_b_and_det_J_b_structured(n_boundaries, n_intervals, x_min, x_max, y_min, y_max, x_G_boundary, weight_G_boundary):

    # x_G_b = []         
    # det_J_b_time_weight = []    # determin of jacobian

    # for i in range(n_boundaries):
    #     for j in range(n_intervals):
    #         if i==0:              # bottom boundary
    #             x_ver_b = np.array([x_min+(x_max-x_min)/n_intervals*j, x_min+(x_max-x_min)/n_intervals*(j+1)])
                
    #             for k in range(len(x_G_boundary)):
    #                 x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2
    #                 y_G_ij_k =y_min
    #                 x_G_b.append([x_G_ij_k, y_G_ij_k])

    #                 det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])
            
    #         if i==1:              # right boundary
    #             y_ver_b = np.array([y_min+(y_max-y_min)/n_intervals*j, y_min+(y_max-y_min)/n_intervals*(j+1)])

    #             for k in range(len(x_G_boundary)):
    #                 x_G_ij_k = x_max
    #                 y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
    #                 x_G_b.append([x_G_ij_k, y_G_ij_k])

    #                 det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])
            
            
    #             """
    #             since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
    #             for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jacobian term
    #             if we integral from left to right, ds = dx
    #             """
    #         if i==2:              # top boundary
    #             x_ver_b = np.array([x_min+(x_max-x_min)/n_intervals*j, x_min+(x_max-x_min)/n_intervals*(j+1)])
    #             # if x_ver_b = np.array([x_max-(x_max-x_min)/n_intervals*j, x_max-(x_max-x_min)/n_intervals*(j+1)]), we integral from right to left, det_J_b_time_weight should be -((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])

    #             for k in range(len(x_G_boundary)):
    #                 x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2  # if 
    #                 y_G_ij_k =y_max
    #                 x_G_b.append([x_G_ij_k, y_G_ij_k])

    #                 det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])
            
    #             """
    #             since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
    #             for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
    #             """
    #         if i==3:              # left boundary
    #             y_ver_b = np.array([y_min+(y_max-y_min)/n_intervals*j, y_min+(y_max-y_min)/n_intervals*(j+1)])
    #             #if y_ver_b = np.array([y_max-(y_max-y_min)/n_intervals*j, y_max-(y_max-y_min)/n_intervals*(j+1)]), we integral from top to bottom, det_J_b_time_weight should be -((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

    #             for k in range(len(x_G_boundary)):
    #                 x_G_ij_k = x_min
    #                 y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
    #                 x_G_b.append([x_G_ij_k, y_G_ij_k])

    #                 det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])
    # return x_G_b, det_J_b_time_weight

    # Cast all to float64 arrays
    x_G_boundary = np.array(x_G_boundary, dtype=np.float64)
    weight_G_boundary = np.array(weight_G_boundary, dtype=np.float64)

    n_gauss = x_G_boundary.shape[0]
    # Prepare for each boundary:
    # For bottom/top: intervals in x, y is constant
    # For right/left: intervals in y, x is constant

    # Precompute interval edges for x, y
    x_edges = np.linspace(x_min, x_max, n_intervals + 1, dtype=np.float64)
    y_edges = np.linspace(y_min, y_max, n_intervals + 1, dtype=np.float64)

    # Vectorized midpoints and half-lengths for all intervals (shape [n_intervals])
    x_mids = (x_edges[:-1] + x_edges[1:]) / 2.0
    x_halfs = (x_edges[1:] - x_edges[:-1]) / 2.0
    y_mids = (y_edges[:-1] + y_edges[1:]) / 2.0
    y_halfs = (y_edges[1:] - y_edges[:-1]) / 2.0

    # For vectorization: expand into shape (n_intervals, n_gauss)
    # Used below for all boundaries

    # --- Bottom boundary (i=0): y = y_min, mapped x from intervals
    xb_bot = (
        x_halfs[:, None] * x_G_boundary[None, :] + x_mids[:, None]
    )  # (n_intervals, n_gauss)
    yb_bot = np.full_like(xb_bot, y_min)
    detJw_bot = (x_halfs[:, None] * weight_G_boundary[None, :])  # (n_intervals, n_gauss)

    # --- Right boundary (i=1): x = x_max, mapped y from intervals
    xr_rit = np.full((n_intervals, n_gauss), x_max, dtype=np.float64)
    yr_rit = (
        y_halfs[:, None] * x_G_boundary[None, :] + y_mids[:, None]
    )  # (n_intervals, n_gauss)
    detJw_rit = (y_halfs[:, None] * weight_G_boundary[None, :])

    # --- Top boundary (i=2): y = y_max, mapped x from intervals
    xt_top = (
        x_halfs[:, None] * x_G_boundary[None, :] + x_mids[:, None]
    )
    yt_top = np.full_like(xt_top, y_max)
    detJw_top = (x_halfs[:, None] * weight_G_boundary[None, :])

    # --- Left boundary (i=3): x = x_min, mapped y from intervals
    xl_lft = np.full((n_intervals, n_gauss), x_min, dtype=np.float64)
    yl_lft = (
        y_halfs[:, None] * x_G_boundary[None, :] + y_mids[:, None]
    )
    detJw_lft = (y_halfs[:, None] * weight_G_boundary[None, :])

    # Concatenate all boundaries in (boundary, n_intervals, n_gauss, 2)
    xys = [
        np.stack([xb_bot, yb_bot], axis=-1),  # (n_intervals, n_gauss, 2)
        np.stack([xr_rit, yr_rit], axis=-1),
        np.stack([xt_top, yt_top], axis=-1),
        np.stack([xl_lft, yl_lft], axis=-1),
    ]
    det_Js = [
        detJw_bot,
        detJw_rit,
        detJw_top,
        detJw_lft,
    ]

    # Stack along new axis, then reshape to (n_boundaries * n_intervals * n_gauss, 2)
    xys_arr = np.concatenate([arr.reshape(-1, 2) for arr in xys], axis=0)
    detJ_arr = np.concatenate([j.reshape(-1) for j in det_Js], axis=0)

    return xys_arr, detJ_arr

# @njit
def x_G_and_def_J_time_weight_multi_grains(num_of_cell,x_G_domain_rec, x_G_domain_tri,weight_G_domain_rec, weight_G_domain_tri, cell_shape, cell_nodes_list, grain_id, angle, repeated_vertex):
    # gauss_angle = [] # corresponding angle of each gauss point
    # x_G = []      # xy coordinates of gauss points in domain   
    # Gauss_grain_id = []
    # det_J_time_weight = []    # determin of jacobian
    # for i in range(num_of_cell):

    #     if cell_shape[i] == 'rec':
        
    #         # in the ith cell calculate get xy coordinates of each domain vertex
    #         x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0], cell_nodes_list[i][3][0]],dtype=np.float64)
    #         y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1], cell_nodes_list[i][3][1]],dtype=np.float64)
    #         # calculate the cy coordinates of gauss points in current integration domain
    #         for k in range(len(x_G_domain_rec)):
    #             gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
    #             x_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
    #                                     (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(x_ver_mn))
    #             y_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
    #                                     (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(y_ver_mn))
    #             x_G.append([x_G_mn_k, y_G_mn_k])
    #             Gauss_grain_id.append(grain_id[i])
    #             J1 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(x_ver_mn))
    #             J2 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(y_ver_mn))
    #             J3 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(x_ver_mn))
    #             J4 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(y_ver_mn))

    #             det_J_time_weight.append(np.linalg.det(np.array([[J1, J2],[J3,J4]]))*weight_G_domain_rec[k])  # each gauss point belongs to same cell has same jacobian
        
    #     if cell_shape[i] == 'tri':
            
    #         # # in the ith cell calculate get xy coordinates of each domain vertex
    #         # x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0]],dtype=np.float64)
    #         # y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1]],dtype=np.float64)
    #         # # calculate the cy coordinates of gauss points in current integration domain
    #         # for k in range(len(x_G_domain_tri)):
    #         #     gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
    #         #     x_G_mn_k = cell_nodes_list[i][0][0] + (cell_nodes_list[i][1][0]-cell_nodes_list[i][0][0])*x_G_domain_tri[k][0] + (cell_nodes_list[i][2][0]-cell_nodes_list[i][0][0])*x_G_domain_tri[k][1]
    #         #     y_G_mn_k = cell_nodes_list[i][0][1] + (cell_nodes_list[i][1][1]-cell_nodes_list[i][0][1])*x_G_domain_tri[k][0] + (cell_nodes_list[i][2][1]-cell_nodes_list[i][0][1])*x_G_domain_tri[k][1]
    #         #     x_G.append([x_G_mn_k, y_G_mn_k])
    #         #     J1 = cell_nodes_list[i][1][0] - cell_nodes_list[i][0][0]
    #         #     J2 = cell_nodes_list[i][1][1] - cell_nodes_list[i][0][1]
    #         #     J3 = cell_nodes_list[i][2][0] - cell_nodes_list[i][0][0]
    #         #     J4 = cell_nodes_list[i][2][1] - cell_nodes_list[i][0][1]

    #         #     det_J_time_weight.append(np.linalg.det(np.array([[J1, J3],[J2,J4]]))*weight_G_domain_tri[k])  # each gauss point belongs to same cell has same jacobian
            
            
            
    #         # in the ith cell calculate get xy coordinates of each domain vertex
    #         if repeated_vertex[i] == 'first':
    #             x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0]],dtype=np.float64)
    #             y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1]],dtype=np.float64)
    #         if repeated_vertex[i] == 'three':
    #             x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0], cell_nodes_list[i][2][0]],dtype=np.float64)
    #             y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1], cell_nodes_list[i][2][1]],dtype=np.float64)
    #         # calculate the cy coordinates of gauss points in current integration domain
    #         for k in range(len(x_G_domain_rec)):
    #             gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
    #             x_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
    #                                     (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(x_ver_mn))
    #             y_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
    #                                     (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(y_ver_mn))
    #             x_G.append([x_G_mn_k, y_G_mn_k])
    #             Gauss_grain_id.append(grain_id[i])
    #             # if i == 3298:
    #             # print(x_G_mn_k, y_G_mn_k)
    #             J1 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(x_ver_mn))
    #             J2 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(y_ver_mn))
    #             J3 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(x_ver_mn))
    #             J4 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(y_ver_mn))
    #             # det_J_time_weight.append(5.0e-15)
    #             det_J_time_weight.append(np.linalg.det(np.array([[J1, J2],[J3,J4]]))*weight_G_domain_rec[k])  # each gauss point belongs to same cell has same jacobian
    # return x_G, det_J_time_weight, gauss_angle, Gauss_grain_id
   
    # Ensure inputs are correct types
    n_cells = int(num_of_cell)
    if n_cells != len(cell_shape) or n_cells != len(cell_nodes_list) or n_cells != len(grain_id):
        raise ValueError("Input lists must match num_of_cell length.")

    # Number of Gauss points per cell (rectangular domain points used for both shapes per reference)
    # Convert to cuPyNumeric arrays for computation
    G_rec = np.array(x_G_domain_rec, dtype=np.float64)  # shape (nGP, 2)
    # Note: x_G_domain_tri is currently unused in reference implementation
    w_rec = np.array(weight_G_domain_rec, dtype=np.float64)
    nGP = int(G_rec.shape[0])

    # Extract xi and eta for Gauss points
    xi = G_rec[:, 0]
    eta = G_rec[:, 1]

    # Build shape function matrix N (nGP x 4) and derivatives dN_dxi, dN_deta
    # Following the exact formulas from the reference implementation
    one = np.array(1.0, dtype=np.float64)
    quarter = np.array(0.25, dtype=np.float64)

    N = np.empty((nGP, 4), dtype=np.float64)
    N[:, 0] = (one - xi) * (one - eta) * quarter
    N[:, 1] = (one + xi) * (one - eta) * quarter
    N[:, 2] = (one + xi) * (one + eta) * quarter
    N[:, 3] = (one - xi) * (one + eta) * quarter

    dN_dxi = np.empty((nGP, 4), dtype=np.float64)
    dN_dxi[:, 0] = -(one - eta) * quarter
    dN_dxi[:, 1] = (one - eta) * quarter
    dN_dxi[:, 2] = (one + eta) * quarter
    dN_dxi[:, 3] = (-one - eta) * quarter  # equals -(1 + eta)

    dN_deta = np.empty((nGP, 4), dtype=np.float64)
    dN_deta[:, 0] = -(one - xi) * quarter
    dN_deta[:, 1] = (-one - xi) * quarter  # equals -(1 + xi)
    dN_deta[:, 2] = (one + xi) * quarter
    dN_deta[:, 3] = (one - xi) * quarter

    # Build per-cell vertex arrays (x_ver_all, y_ver_all), each of shape (n_cells, 4)
    # - For rectangles: [v0, v1, v2, v3]
    # - For triangles with repeated vertex:
    #     'first': [v0, v0, v1, v2]
    #     'three': [v0, v1, v2, v2]
    x_ver_all = np.empty((n_cells, 4), dtype=np.float64)
    y_ver_all = np.empty((n_cells, 4), dtype=np.float64)
    # Using a single Python loop to populate the per-cell arrays (input-dependent), computation will be vectorized later.
    for i in range(n_cells):
        shape_i = cell_shape[i]
        nodes = cell_nodes_list[i]
        if shape_i == "rec":
            # Expect 4 nodes
            x_ver_all[i, 0] = float(nodes[0][0])
            x_ver_all[i, 1] = float(nodes[1][0])
            x_ver_all[i, 2] = float(nodes[2][0])
            x_ver_all[i, 3] = float(nodes[3][0])

            y_ver_all[i, 0] = float(nodes[0][1])
            y_ver_all[i, 1] = float(nodes[1][1])
            y_ver_all[i, 2] = float(nodes[2][1])
            y_ver_all[i, 3] = float(nodes[3][1])
        elif shape_i == "tri":
            # Expect 3 nodes; repeat based on repeated_vertex
            rv = repeated_vertex[i]
            if rv == "first":
                x_ver_all[i, 0] = float(nodes[0][0])
                x_ver_all[i, 1] = float(nodes[0][0])
                x_ver_all[i, 2] = float(nodes[1][0])
                x_ver_all[i, 3] = float(nodes[2][0])

                y_ver_all[i, 0] = float(nodes[0][1])
                y_ver_all[i, 1] = float(nodes[0][1])
                y_ver_all[i, 2] = float(nodes[1][1])
                y_ver_all[i, 3] = float(nodes[2][1])
            elif rv == "three":
                x_ver_all[i, 0] = float(nodes[0][0])
                x_ver_all[i, 1] = float(nodes[1][0])
                x_ver_all[i, 2] = float(nodes[2][0])
                x_ver_all[i, 3] = float(nodes[2][0])

                y_ver_all[i, 0] = float(nodes[0][1])
                y_ver_all[i, 1] = float(nodes[1][1])
                y_ver_all[i, 2] = float(nodes[2][1])
                y_ver_all[i, 3] = float(nodes[2][1])
            else:
                raise ValueError(f"Invalid repeated_vertex flag '{rv}' for triangular cell at index {i}")
        else:
            raise ValueError(f"Invalid cell shape '{shape_i}' at index {i}")

    # Compute x and y coordinates for all Gauss points and all cells:
    # N: (nGP, 4), x_ver_all.T: (4, n_cells) -> (nGP, n_cells)
    x_all = N @ x_ver_all.T
    y_all = N @ y_ver_all.T

    # Compute Jacobian entries using derivatives
    J1 = dN_dxi @ x_ver_all.T
    J2 = dN_dxi @ y_ver_all.T
    J3 = dN_deta @ x_ver_all.T
    J4 = dN_deta @ y_ver_all.T

    # Determinant of Jacobian for each Gauss point and cell
    detJ = J1 * J4 - J2 * J3  # shape (nGP, n_cells)

    # Multiply by Gauss weights (broadcast weights along cells)
    detJ_w = detJ * w_rec[:, None]  # shape (nGP, n_cells)

    # Angle lookup: angle list is [id0, angle0, id1, angle1, ...]
    ids = np.array([int(angle[i]) for i in range(0, len(angle), 2)], dtype=np.int64)
    vals = np.array([float(angle[i + 1]) for i in range(0, len(angle), 2)], dtype=np.float64)
    grain_ids_arr = np.array([int(g) for g in grain_id], dtype=np.int64)

    # Vectorized mapping: match each grain_id to corresponding id in 'ids'
    # match: (n_cells, n_ids) boolean -> cast to float and matmul with 'vals' -> (n_cells,)
    match = (grain_ids_arr[:, None] == ids[None, :])
    angles_cell = np.matmul(match.astype(np.float64), vals)

    # Flatten outputs in cell-major order to match reference:
    # The reference loops over cell i, then over Gauss points k. Thus we transpose to (n_cells, nGP) and then reshape.
    x_all_T = x_all.T  # (n_cells, nGP)
    y_all_T = y_all.T  # (n_cells, nGP)
    x_flat = np.reshape(x_all_T, (n_cells * nGP,))
    y_flat = np.reshape(y_all_T, (n_cells * nGP,))

    x_G_out = np.empty((n_cells * nGP, 2), dtype=np.float64)
    x_G_out[:, 0] = x_flat
    x_G_out[:, 1] = y_flat

    detJ_w_T = detJ_w.T  # (n_cells, nGP)
    detJ_w_flat = np.reshape(detJ_w_T, (n_cells * nGP,))

    # Repeat angles and grain IDs across Gauss points per cell
    gauss_angle_out = np.repeat(angles_cell, nGP).astype(np.float64)
    Gauss_grain_id_out = np.repeat(grain_ids_arr, nGP).astype(np.int64)

    # Convert to NumPy arrays for output (as required)
    x_G_np = np.asarray(x_G_out)
    detJ_w_np = np.asarray(detJ_w_flat)
    gauss_angle_np = np.asarray(gauss_angle_out)
    Gauss_grain_id_np = np.asarray(Gauss_grain_id_out)

    return x_G_np, detJ_w_np, gauss_angle_np, Gauss_grain_id_np
    
    


# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian on boundaries

# @njit
def x_G_b_and_det_J_b_multi_grains(x_min, x_max, y_min, y_max, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, x_G_boundary, weight_G_boundary, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, angle):
    # gauss_angle_b = []
    # x_G_b = []         
    
    # det_J_b_time_weight = []    # determin of jacobian
    # Gauss_b_grain_id = []

    # for j in range(len(bottom_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
    #     x_ver_b = np.array([bottom_boundary_cell_nodes_list[j][0], bottom_boundary_cell_nodes_list[j][1]])
        
    #     for k in range(len(x_G_boundary)):
    #         x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2
    #         y_G_ij_k =y_min
    #         x_G_b.append([x_G_ij_k, y_G_ij_k])
    #         gauss_angle_b.append(angle[angle.index(grain_id_bottom[j])+1])
    #         Gauss_b_grain_id.append(grain_id_bottom[j])

    #         det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])
    
    # for j in range(len(right_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
    #     y_ver_b = np.array([right_boundary_cell_nodes_list[j][0], right_boundary_cell_nodes_list[j][1]])
        
    #     for k in range(len(x_G_boundary)):
    #         x_G_ij_k = x_max
    #         y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
    #         x_G_b.append([x_G_ij_k, y_G_ij_k])
    #         Gauss_b_grain_id.append(grain_id_right[j])
    #         gauss_angle_b.append(angle[angle.index(grain_id_right[j])+1])
    #         det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

    # for j in range(len(top_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
    #     x_ver_b = np.array([top_boundary_cell_nodes_list[j][0], top_boundary_cell_nodes_list[j][1]])
        
    #     for k in range(len(x_G_boundary)):
    #         x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2  # if 
    #         y_G_ij_k =y_max
    #         x_G_b.append([x_G_ij_k, y_G_ij_k])
    #         gauss_angle_b.append(angle[angle.index(grain_id_top[j])+1])
    #         Gauss_b_grain_id.append(grain_id_top[j])
    #         det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])

    # for j in range(len(left_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
    #     y_ver_b = np.array([left_boundary_cell_nodes_list[j][0], left_boundary_cell_nodes_list[j][1]])
        
    #     for k in range(len(x_G_boundary)):
    #         x_G_ij_k = x_min
    #         y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
    #         x_G_b.append([x_G_ij_k, y_G_ij_k])
    #         gauss_angle_b.append(angle[angle.index(grain_id_left[j])+1])
    #         Gauss_b_grain_id.append(grain_id_left[j])
    #         det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

    
            
    #         """
    #         since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
    #         for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jaobian term
    #         if we integral from left to right, ds = dx
    #         """
        
        
    #         """
    #         since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
    #         for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
    #         """
            
    # return x_G_b, det_J_b_time_weight, gauss_angle_b, Gauss_b_grain_id

    # Convert Gauss points and weights to cuPyNumeric arrays (float64)
    xG = np.array(x_G_boundary, dtype=np.float64)
    wG = np.array(weight_G_boundary, dtype=np.float64)

    # Build angle mapping arrays: ids_map and angles_map
    # angle is a flattened Python list [id0, angle0, id1, angle1, ...]
    ids_map = np.array(angle[::2], dtype=np.int64)
    angles_map = np.array(angle[1::2], dtype=np.float64)

    # Helper: map grain IDs to angles using vectorized matching
    def map_ids_to_angles(ids_to_map):
        # ids_to_map: np array int64 (shape (n_segments,))
        if ids_map.size == 0:
            # No mapping available; produce zeros
            return np.zeros_like(ids_to_map, dtype=np.float64)

        # Sort ids_map for fast searchsorted
        sort_idx = np.argsort(ids_map)
        ids_sorted = ids_map[sort_idx]
        angles_sorted = angles_map[sort_idx]

        # Positions from searchsorted
        pos = np.searchsorted(ids_sorted, ids_to_map)
        # Clamp pos to valid range for safety
        pos = np.clip(pos, 0, ids_sorted.size - 1)

        # Candidate angles
        angle_values = angles_sorted[pos]
        matched = (ids_sorted[pos] == ids_to_map)

        # If some IDs were not matched exactly, fallback to broadcast equality
        # This ensures correctness even with unsorted or missing ids.
        if bool(np.any(~matched)):
            eq = (ids_to_map[:, None] == ids_map[None, :])
            # argmax will return index of first True per row, or 0 if no True
            match_pos = np.argmax(eq, axis=1)
            angle_values_fallback = angles_map[match_pos]
            angle_values = np.where(matched, angle_values, angle_values_fallback)

        return angle_values

    # Helper to process boundaries with segments along x (bottom and top)
    def process_x_segments(segments_list, y_const, grain_ids_list):
        # segments_list: list of [x_start, x_end] per segment
        # y_const: float for constant y on this boundary
        # grain_ids_list: list of grain IDs per segment
        n_segments = len(segments_list)
        if n_segments == 0:
            # Empty arrays for no segments
            return (np.empty((0, 2), dtype=np.float64),
                    np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.int64))
        seg = np.array(segments_list, dtype=np.float64)  # (n_segments, 2)
        # Compute half length and center for each segment
        half_len = (seg[:, 1] - seg[:, 0]) * 0.5         # (n_segments,)
        center = (seg[:, 1] + seg[:, 0]) * 0.5           # (n_segments,)

        # Compute x coordinates for all Gauss points per segment: shape (n_segments, n_gauss)
        n_gauss = xG.size
        # Use broadcasting
        x_vals = half_len[:, None] * xG[None, :] + center[:, None]
        # y is constant for this boundary
        y_vals = np.full((n_segments, n_gauss), y_const, dtype=np.float64)

        # det_J_b_time_weight: half_len * weight per gauss, broadcast
        det_vals = half_len[:, None] * wG[None, :]  # (n_segments, n_gauss)

        # Angle mapping per segment, then repeat for every gauss point
        grain_ids_arr = np.array(grain_ids_list, dtype=np.int64)  # (n_segments,)
        angles_per_segment = map_ids_to_angles(grain_ids_arr)      # (n_segments,)

        # Repeat per gauss point
        gauss_angles = np.repeat(angles_per_segment, n_gauss)     # (n_segments*n_gauss,)
        gauss_grain_ids = np.repeat(grain_ids_arr, n_gauss)       # (n_segments*n_gauss,)

        # Stack coordinates and flatten in segment-major order to match reference
        coords = np.stack([x_vals.reshape(-1), y_vals.reshape(-1)], axis=1)  # (n_segments*n_gauss, 2)
        det_flat = det_vals.reshape(-1)                                       # (n_segments*n_gauss,)

        return coords, det_flat, gauss_angles, gauss_grain_ids

    # Helper to process boundaries with segments along y (right and left)
    def process_y_segments(segments_list, x_const, grain_ids_list):
        # segments_list: list of [y_start, y_end] per segment
        # x_const: float for constant x on this boundary
        # grain_ids_list: list of grain IDs per segment
        n_segments = len(segments_list)
        if n_segments == 0:
            # Empty arrays for no segments
            return (np.empty((0, 2), dtype=np.float64),
                    np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.int64))
        seg = np.array(segments_list, dtype=np.float64)  # (n_segments, 2)
        # Compute half length and center for each segment
        half_len = (seg[:, 1] - seg[:, 0]) * 0.5         # (n_segments,)
        center = (seg[:, 1] + seg[:, 0]) * 0.5           # (n_segments,)

        # Compute y coordinates for all Gauss points per segment: shape (n_segments, n_gauss)
        n_gauss = xG.size
        y_vals = half_len[:, None] * xG[None, :] + center[:, None]
        # x is constant for this boundary
        x_vals = np.full((n_segments, n_gauss), x_const, dtype=np.float64)

        # det_J_b_time_weight: half_len_y * weight per gauss
        det_vals = half_len[:, None] * wG[None, :]

        # Angle mapping per segment
        grain_ids_arr = np.array(grain_ids_list, dtype=np.int64)
        angles_per_segment = map_ids_to_angles(grain_ids_arr)

        # Repeat per gauss point
        gauss_angles = np.repeat(angles_per_segment, n_gauss)
        gauss_grain_ids = np.repeat(grain_ids_arr, n_gauss)

        # Stack coordinates and flatten
        coords = np.stack([x_vals.reshape(-1), y_vals.reshape(-1)], axis=1)
        det_flat = det_vals.reshape(-1)

        return coords, det_flat, gauss_angles, gauss_grain_ids

    # Process all boundaries in the reference's order: bottom, right, top, left
    coords_bottom, det_bottom, angles_bottom, ids_bottom = process_x_segments(
        bottom_boundary_cell_nodes_list, y_min, grain_id_bottom
    )
    coords_right, det_right, angles_right, ids_right = process_y_segments(
        right_boundary_cell_nodes_list, x_max, grain_id_right
    )
    coords_top, det_top, angles_top, ids_top = process_x_segments(
        top_boundary_cell_nodes_list, y_max, grain_id_top
    )
    coords_left, det_left, angles_left, ids_left = process_y_segments(
        left_boundary_cell_nodes_list, x_min, grain_id_left
    )

    # Concatenate results for all boundaries
    coords_all = np.concatenate([coords_bottom, coords_right, coords_top, coords_left], axis=0)
    det_all = np.concatenate([det_bottom, det_right, det_top, det_left], axis=0)
    angles_all = np.concatenate([angles_bottom, angles_right, angles_top, angles_left], axis=0)
    ids_all = np.concatenate([ids_bottom, ids_right, ids_top, ids_left], axis=0)

    # Return NumPy arrays to match the reference implementation format
    return (
        np.array(coords_all, dtype=np.float64),
        np.array(det_all, dtype=np.float64),
        np.array(angles_all, dtype=np.float64),
        np.array(ids_all, dtype=np.int64),
    )