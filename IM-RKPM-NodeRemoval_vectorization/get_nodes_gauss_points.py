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


@jit(nopython=True)
def get_x_nodes_single_grain(n_nodes,x_min,x_max,n_intervals,y_min,y_max):
    x_nodes = []
    for j in range(n_nodes):
        for i in range(n_nodes):
            x_nodes.append([x_min+(x_max-x_min)/n_intervals*i, y_min+(y_max-y_min)/n_intervals*j])
    return x_nodes

@jit
def get_x_nodes_multi_grain(x_min,x_max,y_min,y_max, num_pixels_x, num_pixels_y, img_):
    # define initial RPK nodes
    x_nodes_ini = []
    for j in range(num_pixels_x):
        for i in range(num_pixels_y):
            x_nodes_ini.append([x_min+(x_max-x_min)/(num_pixels_x-1)*j, y_min+(y_max-y_min)/(num_pixels_y-1)*i])

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

                # cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*j],\
                #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*i, y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                # grain_id.append(img_[i,j])
                # cell_shape.append('rec')
                # num_rec_cell  = num_rec_cell + 1
                
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
                    # if i == 0 and j == num_pixels_y-2:
                    #     print(([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                    #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+0.5), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                    #                     [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+0.5)]]))
                    #     print(len(cell_nodes_list))
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

                    # cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                    #                             [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                    #                             [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                    #                             [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                    # grain_id.append(img_[i,j])
                    # cell_shape.append('rec')
                    # num_rec_cell  = num_rec_cell + 1

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

                # cell_nodes_list.append([[x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                #                                 [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j)],\
                #                                 [x_min+(x_max-x_min)/(num_pixels_x-1)*(i+1), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)],\
                #                                 [x_min+(x_max-x_min)/(num_pixels_x-1)*(i), y_min+(y_max-y_min)/(num_pixels_y-1)*(j+1)]])
                # grain_id.append(img_[i,j])
                # cell_shape.append('rec')
                # num_rec_cell  = num_rec_cell + 1

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

    # print(np.shape(x_nodes), np.shape(x_nodes_ini), np.shape(x_nodes_added))

    # interface_segments
    x_nodes_added = np.array(x_nodes_added)

    return cell_nodes_list, grain_id, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, cell_shape, num_rec_cell, num_tri_cell, x_nodes, nodes_grain_id, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, repeated_vertex, interface_segments, x_nodes_added, x_nodes_added_id


# get all gauss points in domain 
def x_G_and_def_J_time_weight_structured(n_intervals, x_min,x_max,y_min,y_max,x_G_domain,weight_G_domain):
    x_G = []      # xy coordinates of gauss points in domain   
    det_J_time_weight = []    # determin of jacobian
    for n in range(n_intervals):
        for m in range(n_intervals):
            # in the mn (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
            x_ver_mn = np.array([x_min+m*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+m*(x_max-x_min)/n_intervals],dtype=np.float64)
            y_ver_mn = np.array([y_min+n*(y_max-y_min)/n_intervals, y_min+n*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals],dtype=np.float64)
            # calculate the cy coordinates of gauss points in current integration domain
            for k in range(len(x_G_domain)):
                
                x_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain[k][0])*(1-x_G_domain[k][1]), (1+x_G_domain[k][0])*(1-x_G_domain[k][1]), \
                                        (1+x_G_domain[k][0])*(1+x_G_domain[k][1]), (1-x_G_domain[k][0])*(1+x_G_domain[k][1])],dtype=np.float64), np.transpose(x_ver_mn))
                y_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain[k][0])*(1-x_G_domain[k][1]), (1+x_G_domain[k][0])*(1-x_G_domain[k][1]), \
                                        (1+x_G_domain[k][0])*(1+x_G_domain[k][1]), (1-x_G_domain[k][0])*(1+x_G_domain[k][1])],dtype=np.float64), np.transpose(y_ver_mn))
                x_G.append([x_G_mn_k, y_G_mn_k])
                J1 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][1]), (1-x_G_domain[k][1]), (1+x_G_domain[k][1]), (-1-x_G_domain[k][1])]), np.transpose(x_ver_mn))
                J2 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][1]), (1-x_G_domain[k][1]), (1+x_G_domain[k][1]), (-1-x_G_domain[k][1])]), np.transpose(y_ver_mn))
                J3 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][0]), (-1-x_G_domain[k][0]), (1+x_G_domain[k][0]), (1-x_G_domain[k][0])]), np.transpose(x_ver_mn))
                J4 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain[k][0]), (-1-x_G_domain[k][0]), (1+x_G_domain[k][0]), (1-x_G_domain[k][0])]), np.transpose(y_ver_mn))

                det_J_time_weight.append(np.linalg.det(np.array([[J1, J2],[J3,J4]]))*weight_G_domain[k])
                
    return x_G, det_J_time_weight

# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian on boundaries

@jit
def x_G_b_and_det_J_b_structured(n_boundaries, n_intervals, x_min, x_max, y_min, y_max, x_G_boundary, weight_G_boundary):

    x_G_b = []         
    det_J_b_time_weight = []    # determin of jacobian

    for i in range(n_boundaries):
        for j in range(n_intervals):
            if i==0:              # bottom boundary
                x_ver_b = np.array([x_min+(x_max-x_min)/n_intervals*j, x_min+(x_max-x_min)/n_intervals*(j+1)])
                
                for k in range(len(x_G_boundary)):
                    x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2
                    y_G_ij_k =y_min
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])
            
            if i==1:              # right boundary
                y_ver_b = np.array([y_min+(y_max-y_min)/n_intervals*j, y_min+(y_max-y_min)/n_intervals*(j+1)])

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = x_max
                    y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])
            
            
                """
                since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
                for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jacobian term
                if we integral from left to right, ds = dx
                """
            if i==2:              # top boundary
                x_ver_b = np.array([x_min+(x_max-x_min)/n_intervals*j, x_min+(x_max-x_min)/n_intervals*(j+1)])
                # if x_ver_b = np.array([x_max-(x_max-x_min)/n_intervals*j, x_max-(x_max-x_min)/n_intervals*(j+1)]), we integral from right to left, det_J_b_time_weight should be -((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2  # if 
                    y_G_ij_k =y_max
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])
            
                """
                since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
                for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
                """
            if i==3:              # left boundary
                y_ver_b = np.array([y_min+(y_max-y_min)/n_intervals*j, y_min+(y_max-y_min)/n_intervals*(j+1)])
                #if y_ver_b = np.array([y_max-(y_max-y_min)/n_intervals*j, y_max-(y_max-y_min)/n_intervals*(j+1)]), we integral from top to bottom, det_J_b_time_weight should be -((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

                for k in range(len(x_G_boundary)):
                    x_G_ij_k = x_min
                    y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
                    x_G_b.append([x_G_ij_k, y_G_ij_k])

                    det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])
    return x_G_b, det_J_b_time_weight

@njit
def x_G_and_def_J_time_weight_multi_grains(num_of_cell,x_G_domain_rec, x_G_domain_tri,weight_G_domain_rec, weight_G_domain_tri, cell_shape, cell_nodes_list, grain_id, angle, repeated_vertex):
    gauss_angle = [] # corresponding angle of each gauss point
    x_G = []      # xy coordinates of gauss points in domain   
    Gauss_grain_id = []
    det_J_time_weight = []    # determin of jacobian
    for i in range(num_of_cell):

        if cell_shape[i] == 'rec':
        
            # in the ith cell calculate get xy coordinates of each domain vertex
            x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0], cell_nodes_list[i][3][0]],dtype=np.float64)
            y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1], cell_nodes_list[i][3][1]],dtype=np.float64)
            # calculate the cy coordinates of gauss points in current integration domain
            for k in range(len(x_G_domain_rec)):
                gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
                x_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
                                        (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(x_ver_mn))
                y_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
                                        (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(y_ver_mn))
                x_G.append([x_G_mn_k, y_G_mn_k])
                Gauss_grain_id.append(grain_id[i])
                J1 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(x_ver_mn))
                J2 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(y_ver_mn))
                J3 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(x_ver_mn))
                J4 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(y_ver_mn))

                det_J_time_weight.append(np.linalg.det(np.array([[J1, J2],[J3,J4]]))*weight_G_domain_rec[k])  # each gauss point belongs to same cell has same jacobian
        
        if cell_shape[i] == 'tri':
            
            # # in the ith cell calculate get xy coordinates of each domain vertex
            # x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0]],dtype=np.float64)
            # y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1]],dtype=np.float64)
            # # calculate the cy coordinates of gauss points in current integration domain
            # for k in range(len(x_G_domain_tri)):
            #     gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
            #     x_G_mn_k = cell_nodes_list[i][0][0] + (cell_nodes_list[i][1][0]-cell_nodes_list[i][0][0])*x_G_domain_tri[k][0] + (cell_nodes_list[i][2][0]-cell_nodes_list[i][0][0])*x_G_domain_tri[k][1]
            #     y_G_mn_k = cell_nodes_list[i][0][1] + (cell_nodes_list[i][1][1]-cell_nodes_list[i][0][1])*x_G_domain_tri[k][0] + (cell_nodes_list[i][2][1]-cell_nodes_list[i][0][1])*x_G_domain_tri[k][1]
            #     x_G.append([x_G_mn_k, y_G_mn_k])
            #     J1 = cell_nodes_list[i][1][0] - cell_nodes_list[i][0][0]
            #     J2 = cell_nodes_list[i][1][1] - cell_nodes_list[i][0][1]
            #     J3 = cell_nodes_list[i][2][0] - cell_nodes_list[i][0][0]
            #     J4 = cell_nodes_list[i][2][1] - cell_nodes_list[i][0][1]

            #     det_J_time_weight.append(np.linalg.det(np.array([[J1, J3],[J2,J4]]))*weight_G_domain_tri[k])  # each gauss point belongs to same cell has same jacobian
            
            
            
            # in the ith cell calculate get xy coordinates of each domain vertex
            if repeated_vertex[i] == 'first':
                x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0]],dtype=np.float64)
                y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1]],dtype=np.float64)
            if repeated_vertex[i] == 'three':
                x_ver_mn = np.array([cell_nodes_list[i][0][0], cell_nodes_list[i][1][0], cell_nodes_list[i][2][0], cell_nodes_list[i][2][0]],dtype=np.float64)
                y_ver_mn = np.array([cell_nodes_list[i][0][1], cell_nodes_list[i][1][1], cell_nodes_list[i][2][1], cell_nodes_list[i][2][1]],dtype=np.float64)
            # calculate the cy coordinates of gauss points in current integration domain
            for k in range(len(x_G_domain_rec)):
                gauss_angle.append(angle[angle.index(int(grain_id[i]))+1])
                x_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
                                        (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(x_ver_mn))
                y_G_mn_k = 1.0/4.0*np.dot(np.array([(1-x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][0])*(1-x_G_domain_rec[k][1]), \
                                        (1+x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][0])*(1+x_G_domain_rec[k][1])],dtype=np.float64), np.transpose(y_ver_mn))
                x_G.append([x_G_mn_k, y_G_mn_k])
                Gauss_grain_id.append(grain_id[i])
                # if i == 3298:
                # print(x_G_mn_k, y_G_mn_k)
                J1 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(x_ver_mn))
                J2 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][1]), (1-x_G_domain_rec[k][1]), (1+x_G_domain_rec[k][1]), (-1-x_G_domain_rec[k][1])]), np.transpose(y_ver_mn))
                J3 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(x_ver_mn))
                J4 = 1.0/4.0*np.dot(np.array([-(1-x_G_domain_rec[k][0]), (-1-x_G_domain_rec[k][0]), (1+x_G_domain_rec[k][0]), (1-x_G_domain_rec[k][0])]), np.transpose(y_ver_mn))
                # det_J_time_weight.append(5.0e-15)
                det_J_time_weight.append(np.linalg.det(np.array([[J1, J2],[J3,J4]]))*weight_G_domain_rec[k])  # each gauss point belongs to same cell has same jacobian
    return x_G, det_J_time_weight, gauss_angle, Gauss_grain_id


# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian on boundaries

@njit
def x_G_b_and_det_J_b_multi_grains(x_min, x_max, y_min, y_max, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, x_G_boundary, weight_G_boundary, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, angle):
    gauss_angle_b = []
    x_G_b = []         
    
    det_J_b_time_weight = []    # determin of jacobian
    Gauss_b_grain_id = []

    for j in range(len(bottom_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
        x_ver_b = np.array([bottom_boundary_cell_nodes_list[j][0], bottom_boundary_cell_nodes_list[j][1]])
        
        for k in range(len(x_G_boundary)):
            x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2
            y_G_ij_k =y_min
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            gauss_angle_b.append(angle[angle.index(grain_id_bottom[j])+1])
            Gauss_b_grain_id.append(grain_id_bottom[j])

            det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])
    
    for j in range(len(right_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
        y_ver_b = np.array([right_boundary_cell_nodes_list[j][0], right_boundary_cell_nodes_list[j][1]])
        
        for k in range(len(x_G_boundary)):
            x_G_ij_k = x_max
            y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            Gauss_b_grain_id.append(grain_id_right[j])
            gauss_angle_b.append(angle[angle.index(grain_id_right[j])+1])
            det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

    for j in range(len(top_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
        x_ver_b = np.array([top_boundary_cell_nodes_list[j][0], top_boundary_cell_nodes_list[j][1]])
        
        for k in range(len(x_G_boundary)):
            x_G_ij_k = (x_ver_b[1]-x_ver_b[0])/2*x_G_boundary[k]+(x_ver_b[1]+x_ver_b[0])/2  # if 
            y_G_ij_k =y_max
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            gauss_angle_b.append(angle[angle.index(grain_id_top[j])+1])
            Gauss_b_grain_id.append(grain_id_top[j])
            det_J_b_time_weight.append((x_ver_b[1]-x_ver_b[0])/2*weight_G_boundary[k])

    for j in range(len(left_boundary_cell_nodes_list)): # the jth interval on ith bnoundary
        y_ver_b = np.array([left_boundary_cell_nodes_list[j][0], left_boundary_cell_nodes_list[j][1]])
        
        for k in range(len(x_G_boundary)):
            x_G_ij_k = x_min
            y_G_ij_k = (y_ver_b[1]-y_ver_b[0])/2*x_G_boundary[k]+(y_ver_b[1]+y_ver_b[0])/2
            x_G_b.append([x_G_ij_k, y_G_ij_k])
            gauss_angle_b.append(angle[angle.index(grain_id_left[j])+1])
            Gauss_b_grain_id.append(grain_id_left[j])
            det_J_b_time_weight.append((y_ver_b[1]-y_ver_b[0])/2*weight_G_boundary[k])

    
            
            """
            since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
            for top boundary if we integral from right to left, ds=-dx, in this case minus sign should be applied to the boundary integral term. for simplicity we add to negative sign to jaobian term
            if we integral from left to right, ds = dx
            """
        
        
            """
            since the line integral along the boundary is integral of someting times ds where ds is the curve length and it is positive, 
            for left boundary if we integral from top to right, ds=-dy, if we integral from bottom to top, ds = dy
            """
            
    return x_G_b, det_J_b_time_weight, gauss_angle_b, Gauss_b_grain_id