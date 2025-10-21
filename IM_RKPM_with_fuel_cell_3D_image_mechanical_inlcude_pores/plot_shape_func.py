import time
start_time = time.time()
import numpy as np
from numpy import sign

import scipy.sparse as sp

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit, njit, typed
import scipy.sparse as sp

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

from define_buttler_volmer import i_0_complex, alpha_lattice_complex, c_lattice_complex, Dn_complex, ocp_complex, i_se

from get_nodes_gauss_points import get_x_nodes_single_grain, get_x_nodes_multi_grain, x_G_and_def_J_time_weight_structured, x_G_b_and_det_J_b_structured, x_G_and_def_J_time_weight_multi_grains, x_G_b_and_det_J_b_multi_grains

from shape_function_in_domain_plot import compute_phi_M, shape_grad_shape_func

from define_mechanical_stiffness_matrix import mechanical_stiffness_matrix, mechanical_force_matrix, mechanical_C_tensor
from define_diffusion_matrix_form import diffusion_matrix

from define_eva_at_gauss_points import evaluate_at_gauss_points

from read_image import read_in_image

print('Define domain and parameters')
###############################
# Define domain 
###############################
x_min = -10e-6
x_max = 10e-6
y_min = -10e-6
y_max = 10e-6 

###############################
# Define time step
###############################
t = 10.0              # simulate for 10s
nt = 100             # nt is the number of time steps
dt = t/nt            # time step

IM_RKPM = 'True'  # if it is interfacial modified RKPM

###############################
# geometry
###############################

single_grain = 'Faulse'   # True: single grain, Faulse: multiple grains read from an image

if single_grain == 'True':
    angle = 0
else:
    angle = [26.0, np.pi, 75.0, np.pi/4.0, 121.0,np.pi*2.0/3.0, 149.0, np.pi/2.0, 90.0, np.pi/3.0, 81.0, np.pi/4.0, 37.0, np.pi*2.0/3.0, 110.0, 0.0]
    # angle = [26.0, 0.0, 75.0, 0.0, 121.0,0.0, 149.0, 0.0, 90.0, 0.0, 81.0, 0.0, 37.0, 0.0, 110.0, 0.0]

    # angle = {"26":np.pi/6, "75":np.pi/12, "121":np.pi/3, "149":np.pi/6, "90":np.pi/3, "81":np.pi/4, "37":np.pi/12, "110":0} numba doesn't support library, so we are
    # using a list to save the angle of grain 

n_boundaries = 4

###############################
# Define material properties
###############################
Fday = 9.6485e4     # Faraday constant
R = 8.3145e0        # gas constant
Tk = 3.0515e2       # temperature in K

c_max = 49600.0     # maximum concentration

k_con = 10.0        # conductivity

Dx_div_Dy = 100.0

j_applied = -15.0     # j_applied

E = 138.87e9            # Youngs modulus (Pa)
nu = 0.3                # Poisson ratio
lambda_mechanical = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)         # lamme constants

k_i = 0.0125
k_f = 0.015


#######################################
# differential method
#######################################
    
differential_method = 'direct'    # 'implicite' or 'direct'    # specify which differential method to use, implicite: H1, H2, direct: directly differentiate
# if the IM_RKPM=True, differential method must be set to be direct.

integral_method = 'gauss'
damage_model = 'ON'       # ON or OFF

if integral_method == 'gauss':
# Define Guass int points and cells
    x_G_domain_rec = [[-3**0.5/3, -3**0.5/3],[-3**0.5/3, 3**0.5/3],[3**0.5/3, -3**0.5/3],[3**0.5/3, 3**0.5/3]] # coordinates of 2D Gauss points in Neutral coordinate system for square doamin
    x_G_domain_tri = [[1.0/6.0, 2.0/3.0],[1.0/6.0,1.0/6.0],[2.0/3.0, 1.0/6.0]]
    weight_G_domain_rec = [1.0,1.0,1.0,1.0]         # weight of each 2D Gauss points for rectangular
    weight_G_domain_tri = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    x_G_boundary = [-(3.0/7.0+2.0/7.0*(1.2)**0.5)**0.5, -(3.0/7.0-2.0/7.0*(1.2)**0.5)**0.5, (3.0/7.0-2.0/7.0*(1.2)**0.5)**0.5, (3.0/7.0+2.0/7.0*(1.2)**0.5)**0.5]#[-0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0,0.4058451513773972,0.7415311855993945,0.9491079123427585]#                   # coordinates of 1D Gauss points
    weight_G_boundary = [0.5-30**0.5/36, 0.5+30**0.5/36, 0.5+30**0.5/36, 0.5-30**0.5/36]#[0.1294849661688697,0.2797053914892766,0.3818300505051189,0.4179591836734694,0.3818300505051189,0.2797053914892766,0.1294849661688697]#         # weight of each 1D Gauss points

def_para_time = time.time()

print('time to define parameters = ' + "%s seconds" % (def_para_time - start_time))

############################################
# define RK nodes and gauss points(if needed)
############################################
print('define RK nodes')

if single_grain == 'True':
    n_intervals = 20      # number of intervals along each direction
    n_nodes = n_intervals+1 # number of nodes along each direction,
    x_nodes = np.array(get_x_nodes_single_grain(n_nodes,x_min,x_max,n_intervals,y_min,y_max)) # types: array
    num_interface_segments = 0
    interface_nodes = []
    BxByCxCy = []
else:
    file_name = 'reduced_8grain_ulm_large_grain_51x51.tiff'
    img_, unic_grain_id, num_pixels_x, num_pixels_y = read_in_image(file_name)
    # print(num_pixels_x, num_pixels_y)
    cell_nodes_list, grain_id, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, cell_shape, num_rec_cell, num_tri_cell, x_nodes, nodes_grain_id, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, repeated_vertex, interface_segments  = get_x_nodes_multi_grain(x_min,x_max,y_min,y_max, num_pixels_x, num_pixels_y, img_)
    num_interface_segments = np.shape(interface_segments)[0]
    interface_nodes = np.asarray(interface_segments).reshape(num_interface_segments*2, 2)
    BxByCxCy = np.asarray(interface_segments).reshape(num_interface_segments, 4)  # the first column is x coordinates of point B, ......

    # cell_nodes_list: the ith row correspondes to ith domain cell, it includes the coordinates of all nodes that forms this cell
    # grain_id: grain id of each domain cell
    #grain_id_bottom: grain id of bottom boundary cell
    # cell_shape: triangle or rectangle domain cell
    #num_rec_cell: total number of rectangle cell
    #x_nodes: coordinates of all nodes, shape: number of nodes times 2
    # bottom_boundary_cell_nodes_list: coordinates
    # print(len(repeated_vertex))

    boundary_nodes = np.asarray(bottom_boundary_cell_nodes_list+top_boundary_cell_nodes_list+left_boundary_cell_nodes_list+right_boundary_cell_nodes_list)
    
    # x_nodes is array, all others are lists
    num_of_cell = int(num_rec_cell + num_tri_cell)    
    # print('number of rectangular cell: '+str(num_rec_cell)) 
    # print('number of triangular cell: '+str(num_tri_cell)) 

num_nodes = np.shape(x_nodes)[0]
# print(cell_nodes_list[0])

# for i in range(num_interface_segments):
#     plt.plot([BxByCxCy[i, 0], BxByCxCy[i, 2]], [BxByCxCy[i, 1], BxByCxCy[i, 3]], 'bo', linestyle='-')
# plt.show()

# plt.plot(x_nodes[:,0], x_nodes[:,1], 'o')
# plt.show()
# exit()



print('number of nodes: ' + str(num_nodes))
# print('number of cells: ' + str(num_of_cell))

# print('number of nodes on boundaries:  ', len(bottom_boundary_cell_nodes_list)+len(top_boundary_cell_nodes_list)+len(left_boundary_cell_nodes_list)+len(right_boundary_cell_nodes_list))


print('define gauss points')
# compute the xy coordinates of each gauss points in each gauss domain and the Jacobian
if integral_method == 'gauss':
    if single_grain == 'True':
        x_G, det_J_time_weight = x_G_and_def_J_time_weight_structured(n_intervals, x_min,x_max,y_min,y_max,x_G_domain_rec,weight_G_domain_rec)
        x_G_b, det_J_b_time_weight = x_G_b_and_det_J_b_structured(n_boundaries, n_intervals, x_min, x_max, y_min, y_max, x_G_boundary, weight_G_boundary)
        gauss_angle = angle*np.ones(len(x_G))
        gauss_angle_b = angle*np.ones(len(x_G_b))
    else:
        x_G_domain_tri = typed.List([typed.List(x) for x in x_G_domain_tri])
        x_G_domain_rec = typed.List([typed.List(x) for x in x_G_domain_rec])
        cell_nodes_list = typed.List([typed.List(x) for x in cell_nodes_list])

        x_G, det_J_time_weight, gauss_angle, Gauss_grain_id = x_G_and_def_J_time_weight_multi_grains(num_of_cell,x_G_domain_rec, x_G_domain_tri,weight_G_domain_rec, weight_G_domain_tri, cell_shape, cell_nodes_list, grain_id, angle, repeated_vertex)
        bottom_boundary_cell_nodes_list = typed.List([typed.List(x) for x in bottom_boundary_cell_nodes_list])
        right_boundary_cell_nodes_list = typed.List([typed.List(x) for x in right_boundary_cell_nodes_list])
        left_boundary_cell_nodes_list = typed.List([typed.List(x) for x in left_boundary_cell_nodes_list])
        top_boundary_cell_nodes_list = typed.List([typed.List(x) for x in top_boundary_cell_nodes_list])
        x_G_b, det_J_b_time_weight, gauss_angle_b, Gauss_b_grain_id = x_G_b_and_det_J_b_multi_grains(x_min, x_max, y_min, y_max, bottom_boundary_cell_nodes_list, right_boundary_cell_nodes_list, top_boundary_cell_nodes_list, left_boundary_cell_nodes_list, x_G_boundary, weight_G_boundary, grain_id_bottom, grain_id_top, grain_id_left, grain_id_right, angle)
    x_G = np.array(x_G)
    x_G_b = np.array(x_G_b)
    num_gauss_points_in_domain = np.shape(x_G)[0]
    num_gauss_points_on_boundary = np.shape(x_G_b)[0]
    gauss_angle = np.array(gauss_angle)
    gauss_angle_b = np.array(gauss_angle_b)
    Gauss_grain_id = np.array(Gauss_grain_id)
    Gauss_b_grain_id = np.array(Gauss_b_grain_id)

# # print(np.max(det_J_time_weight), np.min(det_J_time_weight), np.average(det_J_time_weight))
# print('number of Gauss points in domain: ' + str(num_gauss_points_in_domain))
# print('number of Gauss points on boundaries: ' + str(num_gauss_points_on_boundary))
# print(gauss_angle[0])
# exit()

# c_bar = plt.scatter(x_G[:,0], x_G[:,1], s=5, c=gauss_angle)
# plt.colorbar(c_bar)
# plt.show()
# exit()

# plt.scatter(x_G[:,0], x_G[:,1], s=8, color = 'hotpink')
# plt.scatter(x_nodes[:,0], x_nodes[:,1], s=8, color='#88c999')
# plt.show()
# exit()

def_nodes_gauss_points_time = time.time()

print('time to define nodes and Gauss points = ' + "%s seconds" % (def_nodes_gauss_points_time-def_para_time))

####################################################
# Compute shape function and its gradient in domain
#####################################################
print('Compute shape function and its gradient in domain')

HT0 = np.array([1,0,0],dtype=np.float64)     # transpose of the basis vector H
HT1 = np.array([0,-1,0],dtype=np.float64)   # for computation of gradient of shape function, d/dx
HT2 = np.array([0,0,-1],dtype=np.float64)   # for computation of gradient of shape function, d/dy

c = 2.0       # support size

if single_grain == 'True':
    a = c*(x_max-x_min)/n_intervals*np.ones(num_nodes)       # compact support size, shape: (num_nodes,)
else:

    h = np.zeros(num_nodes)
    for i in range(num_nodes):
        dist = ((x_nodes[i,0]-x_nodes[:,0])**2+(x_nodes[i,1]-x_nodes[:,1])**2)**0.5
        
        index_four_smallest = sorted(range(len(dist)), key=lambda sub: dist[sub])[:5]  # get the index of the four smallest index, the first one is always zero, so 5 here

        h[i] = dist[index_four_smallest][dist[index_four_smallest].tolist().index(max(dist[index_four_smallest]))]

    a = c*h #shape: (num_nodes,)

# print(h[12*51])
# print(x_nodes[12*51,:])
# print(np.max(a), np.min(a), np.average(a))
# exit()
    

##############################################################
# find two nodes whose shape function will be ploted, the two
# nodes are at different side of an interface
###############################################################5.600000000000000594e-06
    #-4.400000000000000188e-06
node_ID_to_plot_1_coor =   np.array([-4.400000000000000594e-06, -4.000000000000000666e-06])# the coordinates of first node whose shape function will be plotted 
node_ID_to_plot_2_coor =  np.array([-4.400000000000000594e-06, -4.400000000000000224e-06])# the coordinates of second node whose shape function will be plotted
node_ID_to_plot_interface_coor = np.array([-4.400000000000000594e-06, -4.200000000000000445e-06]) # the coordinates of node on interface whose shape function will be plotted

node_ID_to_plot_1_index = np.where(np.linalg.norm(x_nodes - node_ID_to_plot_1_coor, 2, axis = 1) < 1e-10)[0][0]
node_ID_to_plot_2_index = np.where(np.linalg.norm(x_nodes - node_ID_to_plot_2_coor, 2, axis = 1) < 1e-10)[0][0]
node_ID_to_plot_interface_index = np.where(np.linalg.norm(x_nodes - node_ID_to_plot_interface_coor, 2, axis = 1) < 1e-10)[0][0]

# print(node_ID_to_plot_1_index, node_ID_to_plot_2_index, node_ID_to_plot_interface_index)
# print(a[[node_ID_to_plot_1_index, node_ID_to_plot_2_index, node_ID_to_plot_interface_index]])
# print(x_nodes[node_ID_to_plot_1_index, :], x_nodes[node_ID_to_plot_2_index, :], x_nodes[node_ID_to_plot_interface_index, :])
# exit()


# print(node_ID_to_plot_1_index)
# print(x_nodes[node_ID_to_plot_1_index, :])
# print(node_ID_to_plot_2_index)
# print(x_nodes[node_ID_to_plot_2_index, :])
# print(node_ID_to_plot_interface_index)
# print(x_nodes[node_ID_to_plot_interface_index, :])
# np.savetxt('node_coor.txt', x_nodes)
# exit()

######################################################################################################
# Define ploting points (100 points within the circle with center at the node, and radius of 2*H_I)
#####################################################################################################
node_ID_to_plot_1_a = a[node_ID_to_plot_1_index]
node_ID_to_plot_2_a = a[node_ID_to_plot_2_index]
node_ID_to_plot_interface_a = a[node_ID_to_plot_interface_index]

node_1_plotting_points = []
node_2_plotting_points = []
node_interface_plotting_points = []

node_1_plotting_points_grain_id = []
node_2_plotting_points_grain_id = []
node_interface_plotting_points_grain_id = []

num_plot_points_node_1 = 0
num_plot_points_node_2 = 0
num_plot_points_node_interface = 0

# x_1 = []
# y_1 = []

# x_2 = []
# y_2 = []

# x_inter = []
# y_inter = []

# all_plotting_grain_id = []
# all_plotting_points = []

grid_num = 100
for i in range(grid_num+1):
    for j in range(grid_num+1):

        plotting_points_1_x = node_ID_to_plot_1_coor[0] - node_ID_to_plot_1_a + 2*node_ID_to_plot_1_a/grid_num*(i)
        # x_1.append(plotting_points_1_x)
        plotting_points_1_y = node_ID_to_plot_1_coor[1] - node_ID_to_plot_1_a + 2*node_ID_to_plot_1_a/grid_num*(j)
        # y_1.append(plotting_points_1_y)

        plotting_points_2_x = node_ID_to_plot_2_coor[0] - node_ID_to_plot_2_a + 2*node_ID_to_plot_2_a/grid_num*(i)
        # x_2.append(plotting_points_2_x)
        plotting_points_2_y = node_ID_to_plot_2_coor[1] - node_ID_to_plot_2_a + 2*node_ID_to_plot_2_a/grid_num*(j)
        # y_2.append(plotting_points_2_y)

        plotting_points_interface_x = node_ID_to_plot_interface_coor[0] - node_ID_to_plot_interface_a + 2*node_ID_to_plot_interface_a/grid_num*(i)
        # x_inter.append(plotting_points_interface_x)
        plotting_points_interface_y = node_ID_to_plot_interface_coor[1] - node_ID_to_plot_interface_a + 2*node_ID_to_plot_interface_a/grid_num*(j)
        # y_inter.append(plotting_points_interface_y)
        

        if ((plotting_points_1_x-node_ID_to_plot_1_coor[0])**2+(plotting_points_1_y-node_ID_to_plot_1_coor[1])**2)**0.5 < node_ID_to_plot_1_a:
            #remove plotting points on interface, if on interface, the derivative of kernal is infinite
            point_not_on_interface = 'True'

            for i_interface_seg in range(num_interface_segments):
                BP = np.array([plotting_points_1_x-BxByCxCy[i_interface_seg, 0], plotting_points_1_y-BxByCxCy[i_interface_seg, 1]])
                BC = np.array([BxByCxCy[i_interface_seg, 2]-BxByCxCy[i_interface_seg, 0], BxByCxCy[i_interface_seg, 3]-BxByCxCy[i_interface_seg, 1]])
                CP = np.array([plotting_points_1_x-BxByCxCy[i_interface_seg, 2], plotting_points_1_y-BxByCxCy[i_interface_seg, 3]])
                if np.cross(BP, BC) == 0 and np.dot(BP, BC) >=0 and np.dot(BP, BC) <= np.dot(BC, BC):
                    point_not_on_interface = 'False'
                       
            if point_not_on_interface == 'True':
                
                node_1_plotting_points.append([plotting_points_1_x, plotting_points_1_y])
                
                if plotting_points_1_y > node_ID_to_plot_interface_coor[1]:
                    node_1_plotting_points_grain_id.append(nodes_grain_id[node_ID_to_plot_1_index])
                else:
                    node_1_plotting_points_grain_id.append(nodes_grain_id[node_ID_to_plot_2_index])
                
                num_plot_points_node_1 += 1

        if ((plotting_points_2_x-node_ID_to_plot_2_coor[0])**2+(plotting_points_2_y-node_ID_to_plot_2_coor[1])**2)**0.5 < node_ID_to_plot_2_a:
            point_not_on_interface = 'True'

            for i_interface_seg in range(num_interface_segments):
                BP = np.array([plotting_points_2_x-BxByCxCy[i_interface_seg, 0], plotting_points_2_y-BxByCxCy[i_interface_seg, 1]])
                BC = np.array([BxByCxCy[i_interface_seg, 2]-BxByCxCy[i_interface_seg, 0], BxByCxCy[i_interface_seg, 3]-BxByCxCy[i_interface_seg, 1]])
                CP = np.array([plotting_points_2_x-BxByCxCy[i_interface_seg, 2], plotting_points_2_y-BxByCxCy[i_interface_seg, 3]])
                if np.cross(BP, BC) == 0 and np.dot(BP, BC) >=0 and np.dot(BP, BC) <= np.dot(BC, BC):
                    point_not_on_interface = 'False'
                
            
            if point_not_on_interface == 'True':
                node_2_plotting_points.append([plotting_points_2_x, plotting_points_2_y])
                
               
                if plotting_points_2_y > node_ID_to_plot_interface_coor[1]:
                    node_2_plotting_points_grain_id.append(nodes_grain_id[node_ID_to_plot_1_index])
                else:
                    node_2_plotting_points_grain_id.append(nodes_grain_id[node_ID_to_plot_2_index])
                num_plot_points_node_2 += 1


        if ((plotting_points_interface_x-node_ID_to_plot_interface_coor[0])**2+(plotting_points_interface_y-node_ID_to_plot_interface_coor[1])**2)**0.5 <= node_ID_to_plot_interface_a:
            point_not_on_interface = 'True'

            # for i_nodes in range(num_interface_segments*2):
            #     # print('yyy')
            #     if abs(plotting_points_interface_x - interface_nodes[i_nodes, 0])<1e-10 and abs(plotting_points_interface_y - interface_nodes[i_nodes, 1])<1e-10:
            #         point_not_on_interface = 'False'

            for i_interface_seg in range(num_interface_segments):
                BP = np.array([plotting_points_interface_x-BxByCxCy[i_interface_seg, 0], plotting_points_interface_y-BxByCxCy[i_interface_seg, 1]])
                BC = np.array([BxByCxCy[i_interface_seg, 2]-BxByCxCy[i_interface_seg, 0], BxByCxCy[i_interface_seg, 3]-BxByCxCy[i_interface_seg, 1]])
                CP = np.array([plotting_points_interface_x-BxByCxCy[i_interface_seg, 2], plotting_points_interface_y-BxByCxCy[i_interface_seg, 3]])
                if np.cross(BP, BC) == 0 and np.dot(BP, BC) >=0 and np.dot(BP, BC) <= np.dot(BC, BC):
                    point_not_on_interface = 'False'
            
            if point_not_on_interface == 'True':
                
                node_interface_plotting_points.append([plotting_points_interface_x, plotting_points_interface_y])
                if plotting_points_interface_y > node_ID_to_plot_interface_coor[1]:
                    node_interface_plotting_points_grain_id.append(nodes_grain_id[node_ID_to_plot_1_index])
                else:
                    node_interface_plotting_points_grain_id.append(nodes_grain_id[node_ID_to_plot_2_index])
                num_plot_points_node_interface += 1

plotting_points_coor_node_1 = np.array(node_1_plotting_points)
plotting_points_coor_node_2 = np.array(node_2_plotting_points)
plotting_points_coor_node_interface = np.array(node_interface_plotting_points)
plotting_points = np.array([-4.48453e-6, -4.88453e-6]).reshape(1,2)
plotting_grain_id = np.array([nodes_grain_id[node_ID_to_plot_2_index]])
                
all_plotting_points = np.concatenate((plotting_points_coor_node_1, plotting_points_coor_node_2, plotting_points_coor_node_interface, plotting_points), axis=0)  # similarly to x_G

all_plotting_grain_id = np.concatenate((np.array(node_1_plotting_points_grain_id), np.array(node_2_plotting_points_grain_id), np.array(node_interface_plotting_points_grain_id), plotting_grain_id), axis=0)


# all_plotting_points = np.array([-4.48453e-6, -4.88453e-6]).reshape(1,2)
# all_plotting_grain_id = np.array([nodes_grain_id[node_ID_to_plot_2_index]]).reshape(1,1)

# print(all_plotting_grain_id)
# exit()


# np.savetxt('plotting_points_grain_id.txt', all_plotting_grain_id)
# np.savetxt('nodes_grain_id.txt', nodes_grain_id)

# exit()

# plt.plot(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], 'o')
# plt.scatter(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], c=node_interface_plotting_points_grain_id, alpha=0.5)
# plt.show()
# exit()



# all_plotting_points = np.concatenate((plotting_points_coor_node_1, plotting_points_coor_node_2, plotting_points_coor_node_interface), axis=0)  # similarly to x_G

num_all_plotting_points = num_plot_points_node_1 + num_plot_points_node_2 + num_plot_points_node_interface+1

to_be_save = np.zeros((num_all_plotting_points, 3))

to_be_save[:,:2] = all_plotting_points
to_be_save[:,2] = all_plotting_grain_id

np.savetxt('polt_points_grain_id.txt', to_be_save)

# print(nodes_grain_id[node_ID_to_plot_1_index], nodes_grain_id[node_ID_to_plot_2_index])

# exit()
# plt.plot(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], 'o')
# plt.show()
# plt.plot(x_inter, y_inter, 'o')
# plt.plot(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], 'ro')
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.show()
# exit()

# print(num_plot_points_node_1, num_plot_points_node_2, num_plot_points_node_interface)
# print(np.shape(all_plotting_points))


M = np.array([np.zeros((3,3)) for _ in range(num_all_plotting_points)])
M_P_x = np.array([np.zeros((3,3)) for _ in range(num_all_plotting_points)]) # partial M partial x
M_P_y = np.array([np.zeros((3,3)) for _ in range(num_all_plotting_points)]) # partial M partial y

# print(type(x_nodes[1,:]))
# print(type(interface_nodes))

np.savetxt('plotting_points.txt', all_plotting_points)

phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data, phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y, heaviside_list, heaviside_plotting_points = compute_phi_M(all_plotting_points, all_plotting_grain_id, x_nodes,nodes_grain_id,a, M, M_P_x, M_P_y, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM)


heaviside_plotting_points = np.asarray(heaviside_plotting_points)

# fig1 = plt.figure()
# ax = plt.axes(projection ='3d')
# # ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# # ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)
# ax.plot_trisurf(heaviside_plotting_points[:,0], heaviside_plotting_points[:,1], heaviside_list, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, c='red')
# # ax.plot_trisurf(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, c='red')
# plt.show()
# exit()

# numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
phi_a = csc_matrix((np.array(phi_nonzerovalue_data), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_all_plotting_points, num_nodes))

kernal_func_array = phi_a.toarray()
kernal_func_node_1 = kernal_func_array[:num_plot_points_node_1, node_ID_to_plot_1_index]
kernal_func_node_2 = kernal_func_array[num_plot_points_node_1:(num_plot_points_node_1+num_plot_points_node_2), node_ID_to_plot_2_index]
kernal_func_node_interface = kernal_func_array[(num_plot_points_node_1+num_plot_points_node_2):(num_plot_points_node_1+num_plot_points_node_2+num_plot_points_node_interface), node_ID_to_plot_interface_index]

# kernal_func_plotting_point = kernal_func_array[0, [node_ID_to_plot_1_index,node_ID_to_plot_2_index,node_ID_to_plot_interface_index]]

# print(kernal_func_plotting_point)



fig1 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)
ax.plot_trisurf(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], kernal_func_node_1, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, c='red')
ax.plot_trisurf(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], kernal_func_node_2, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, c='red')
# plt.savefig('kernal_not_on_interface.png')
plt.show()

fig2 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)

ax.plot_trisurf(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], kernal_func_node_interface, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], shape_func_value_node_interface, c='red')
# plt.savefig('kernal_on_interface.png')
plt.show()

# # exit()



# phi_a_P_x = csc_matrix((np.array(phi_P_x_nonzerovalue_data), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))
# phi_a_P_y = csc_matrix((np.array(phi_P_y_nonzerovalue_data), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_gauss_points_in_domain, num_nodes))

num_non_zero_phi_a = np.shape(np.array(phi_nonzero_index_row))[0]

shape_func_value,  grad_shape_func_x_value, grad_shape_func_y_value = shape_grad_shape_func(all_plotting_points,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight, IM_RKPM)

# numba doesn't support csc_matrix, so get all these parameters and construct csc_matrix out of numba
shape_func = csc_matrix((np.array(shape_func_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_all_plotting_points, num_nodes))
grad_shape_func_x = csc_matrix((np.array(grad_shape_func_x_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_all_plotting_points, num_nodes))
grad_shape_func_y = csc_matrix((np.array(grad_shape_func_y_value), (np.array(phi_nonzero_index_row),np.array(phi_nonzero_index_column))), shape = (num_all_plotting_points, num_nodes))

comp_shape_func_grad_shape_func_in_domain = time.time()

print('time to compute the shape function and grad of shape function in domain = ' + "%s seconds" % (comp_shape_func_grad_shape_func_in_domain-def_nodes_gauss_points_time))

shape_func_array = shape_func.toarray()

grad_shape_func_x_array = grad_shape_func_x.toarray()
grad_shape_func_y_array = grad_shape_func_y.toarray()

shape_func_value_node_1 = shape_func_array[0:num_plot_points_node_1, node_ID_to_plot_1_index]
shape_func_value_node_2 = shape_func_array[num_plot_points_node_1:(num_plot_points_node_1+num_plot_points_node_2), node_ID_to_plot_2_index]
shape_func_value_node_interface = shape_func_array[(num_plot_points_node_1+num_plot_points_node_2):(num_plot_points_node_1+num_plot_points_node_2+num_plot_points_node_interface), node_ID_to_plot_interface_index]

grad_shape_func_x_node_1 = grad_shape_func_x_array[0:num_plot_points_node_1, node_ID_to_plot_1_index]
grad_shape_func_y_node_1 = grad_shape_func_y_array[0:num_plot_points_node_1, node_ID_to_plot_1_index]
grad_shape_func_x_node_2 = grad_shape_func_x_array[num_plot_points_node_1:(num_plot_points_node_1+num_plot_points_node_2), node_ID_to_plot_2_index]
grad_shape_func_y_node_2 = grad_shape_func_y_array[num_plot_points_node_1:(num_plot_points_node_1+num_plot_points_node_2), node_ID_to_plot_2_index]
grad_shape_func_x_node_interface = grad_shape_func_x_array[(num_plot_points_node_1+num_plot_points_node_2):(num_plot_points_node_1+num_plot_points_node_2+num_plot_points_node_interface), node_ID_to_plot_interface_index]
grad_shape_func_y_node_interface = grad_shape_func_y_array[(num_plot_points_node_1+num_plot_points_node_2):(num_plot_points_node_1+num_plot_points_node_2+num_plot_points_node_interface), node_ID_to_plot_interface_index]


################################
# check reproducing condition
################################
print(np.sum(shape_func_array, axis=1))
print(np.sum(grad_shape_func_x_array, axis=1))
print(np.sum(grad_shape_func_x_array, axis=1))
# print(np.sum(shape_func_array))



# shape_func_value_node_1 = shape_func_array[0, node_ID_to_plot_1_index]
# shape_func_value_node_2 = shape_func_array[0, node_ID_to_plot_2_index]
# shape_func_value_node_interface = shape_func_array[0, node_ID_to_plot_interface_index]

# print(shape_func_value_node_interface)

# grad_shape_func_x_node_1 = grad_shape_func_x_array[0, node_ID_to_plot_1_index]
# grad_shape_func_y_node_1 = grad_shape_func_y_array[0, node_ID_to_plot_1_index]
# grad_shape_func_x_node_2 = grad_shape_func_x_array[0, node_ID_to_plot_2_index]
# grad_shape_func_y_node_2 = grad_shape_func_y_array[0, node_ID_to_plot_2_index]
# grad_shape_func_x_node_interface = grad_shape_func_x_array[0, node_ID_to_plot_interface_index]
# grad_shape_func_y_node_interface = grad_shape_func_y_array[0, node_ID_to_plot_interface_index]
# print(grad_shape_func_x_node_interface, grad_shape_func_y_node_interface)


# print(np.shape(grad_shape_func_x_node_1))
# print(np.shape(plotting_points_coor_node_interface[:,0]))
# print(np.shape(plotting_points_coor_node_interface[:,1]))

# coor_1_x = []
# coor_1_y = []
# shape_node1 = []

# for i in range(np.shape(shape_func_value_node_1)[0]):
#     if shape_func_value_node_1[i] != 0:
#         coor_1_x.append(plotting_points_coor_node_1[i,0])
#         coor_1_y.append(plotting_points_coor_node_1[i,1])
#         shape_node1.append(shape_func_value_node_1[i])

# coor_2_x = []
# coor_2_y = []
# shape_node2 = []

# for i in range(np.shape(shape_func_value_node_2)[0]):
#     if shape_func_value_node_2[i] != 0:
#         coor_2_x.append(plotting_points_coor_node_2[i,0])
#         coor_2_y.append(plotting_points_coor_node_2[i,1])
#         shape_node2.append(shape_func_value_node_2[i])

# fig1 = plt.figure()
# ax = plt.axes(projection ='3d')
# # ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# # ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)
# # ax.plot_trisurf(coor_1_x, coor_1_y, shape_node1, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, c='red')
# # ax.plot_trisurf(coor_2_x, coor_2_y, shape_node2, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, c='red')
# plt.show()


fig1 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)
ax.plot_trisurf(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, c='red')
ax.plot_trisurf(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, c='red')
# plt.savefig('shpaefunc_on_interface.png')
plt.show()

# fig2 = plt.figure()
# ax = plt.axes(projection ='3d')
# # ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# # ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)
# ax.plot_trisurf(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1, c='red')
# # ax.plot_trisurf(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, color='white', edgecolors='grey', alpha=0.5)
# # ax.scatter(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2, c='red')
# plt.show()

fig3 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)

ax.plot_trisurf(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], shape_func_value_node_interface, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], shape_func_value_node_interface, c='red')
# plt.savefig('shape_func_on_interface.png')
plt.show()



fig4 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)

ax.plot_trisurf(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], grad_shape_func_x_node_1, color='white', edgecolors='grey', alpha=0.5)
ax.plot_trisurf(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], grad_shape_func_x_node_2, color='white', edgecolors='grey', alpha=0.5)
# plt.savefig('grad_x_shape_func_not_on_interface.png')
plt.show()

fig5 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)

ax.plot_trisurf(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], grad_shape_func_y_node_1, color='white', edgecolors='grey', alpha=0.5)
ax.plot_trisurf(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], grad_shape_func_y_node_2, color='white', edgecolors='grey', alpha=0.5)
# plt.savefig('grad_y_shape_func_not_on_interface.png')
plt.show()

fig6 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)

ax.plot_trisurf(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], grad_shape_func_x_node_interface, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], shape_func_value_node_interface, c='red')
# plt.savefig('grad_x_shape_func_on_interface.png')
plt.show()

fig7 = plt.figure()
ax = plt.axes(projection ='3d')
# ax.plot3D(plotting_points_coor_node_1[:,0], plotting_points_coor_node_1[:,1], shape_func_value_node_1)
# ax.plot3D(plotting_points_coor_node_2[:,0], plotting_points_coor_node_2[:,1], shape_func_value_node_2)

ax.plot_trisurf(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], grad_shape_func_y_node_interface, color='white', edgecolors='grey', alpha=0.5)
# ax.scatter(plotting_points_coor_node_interface[:,0], plotting_points_coor_node_interface[:,1], shape_func_value_node_interface, c='red')
# plt.savefig('grad_y_shape_func_on_interface.png')
plt.show()