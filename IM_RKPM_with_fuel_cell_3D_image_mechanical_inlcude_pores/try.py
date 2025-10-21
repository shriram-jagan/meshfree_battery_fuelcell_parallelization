import time
start_time = time.time()
import numpy as np
from numpy import sign

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit, njit
import numba
from numba.typed import List
from numba.types import ListType
import scipy.sparse as sp

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

import pickle
import os
import sys
import warnings

electrode_phi_meshfree = [0.04238164255329723, 0.059336592964299904, 0.06974191187933286, 0.0768496665163377, 0.08191337191955059, 0.08554413434585477, 0.0880783984594262, 0.08975793653232227, 0.09069047673835208, 0.09097295144055813]
electrode_phi_comsol = [0.04239976860930609,0.05933580358603316,0.06973989737186347,0.0768531636158785,0.08192057237367924,\
                        0.08555521698620203,0.08810211080116774,0.08977297301875978,0.09070664915755189,0.09100036972411722]

portion = [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
electrolyte_phi_meshfree = [0.34971627597311644, 0.32959998394537027, 0.31725451024848356, 0.30882144441270687, 0.30281355342643407, 0.29850578970501074, 0.2954989778380446, 0.2935062662927766, 0.29239984041452316, 0.29206469403526564]
electrolyte_phi_comsol = [0.3496952622985523,0.3296021990755289,0.317259211753236,0.3088205642981656,0.3028090111266762,\
                          0.2984971280242831,0.29547561451492005,0.29349333545437056,0.292385614441175,0.29203713617920785]

fig1 = plt.figure( )
plt.plot(portion,electrolyte_phi_meshfree, '-ob', label='Meshfree')
plt.plot(portion,electrolyte_phi_comsol, '-or', label='Comsol')
plt.plot([0], [0.29357795249561125], 'xb', label = 'Line Source Meshfree')
plt.plot([0], [0.2863391381371657], 'xr', label = 'Line Source Comsol')
plt.legend()
plt.grid()
plt.xlabel('Interface portion')
plt.ylabel('Max phi in electrolyte')

fig2 = plt.figure( )
plt.plot(portion,electrode_phi_meshfree, '-ob', label='Meshfree')
plt.plot(portion,electrode_phi_comsol, '-or', label='Comsol')

plt.plot([0], [0.08969804434334647], 'xb', label = 'Line Source Meshfree')
plt.plot([0], [0.09580264355852622], 'xr', label = 'Line Source Comsol')
plt.legend()
plt.grid()
plt.xlabel('Interface portion')
plt.ylabel('Min phi in electrode')

plt.show()

exit()


potential_in_domain_save_electrolyte = np.loadtxt('potential_in_domain_electrolyte.txt')
potential_in_domain_save_electrode = np.loadtxt('potential_in_domain_electrode.txt')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
sc = ax1.scatter(potential_in_domain_save_electrolyte[:, 0], potential_in_domain_save_electrolyte[:, 1],potential_in_domain_save_electrolyte[:, 2], c=potential_in_domain_save_electrolyte[:, 3])
# plt.scatter(potential_on_boundary_save_electrolyte[:, 0], potential_on_boundary_save_electrolyte[:, 1],potential_on_boundary_save_electrolyte[:, 2], c=potential_on_boundary_save_electrolyte[:, 3])

plt.colorbar(sc, ax=ax1)

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
sc = ax.scatter(potential_in_domain_save_electrode[:, 0], potential_in_domain_save_electrode[:, 1], potential_in_domain_save_electrode[:, 2], c=potential_in_domain_save_electrode[:, 3])
# plt.scatter(potential_on_boundary_save_electrode[:, 0], potential_on_boundary_save_electrode[:, 1],potential_on_boundary_save_electrode[:, 2], c=potential_on_boundary_save_electrode[:, 3])

plt.colorbar(sc, ax=ax)
plt.show()
# x_min = 0
# x_max = 10.0
# y_min = 0
# y_max = 10.0
# z_min = 0
# z_max = 10.0

# m = 0
# n = 0
# mn = 0

# n_intervals = 1

# x_G_domain = np.array([[-1,-1,-1],[-1,1,-1],[1,-1,-1],[1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1]])
# x_G = []

# # in the mnnm (n^th row, m^th column) gauss integration domain, calculate the xy coordinates of each domain vertex
# x_ver_mn = np.array([x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+(m)*(x_max-x_min)/n_intervals, x_min+m*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+(m+1)*(x_max-x_min)/n_intervals, x_min+(m)*(x_max-x_min)/n_intervals, x_min+m*(x_max-x_min)/n_intervals],dtype=np.float64)
# y_ver_mn = np.array([y_min+n*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals, y_min+(n)*(y_max-y_min)/n_intervals,y_min+n*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals, y_min+(n+1)*(y_max-y_min)/n_intervals, y_min+(n)*(y_max-y_min)/n_intervals],dtype=np.float64)
# z_ver_mn = np.array([z_min+n*(z_max-z_min)/n_intervals, z_min+n*(z_max-z_min)/n_intervals, z_min+n*(z_max-z_min)/n_intervals, z_min+n*(z_max-z_min)/n_intervals, z_min+(n+1)*(z_max-z_min)/n_intervals,z_min+(n+1)*(z_max-z_min)/n_intervals,z_min+(n+1)*(z_max-z_min)/n_intervals,z_min+(n+1)*(z_max-z_min)/n_intervals])
# # calculate the cy coordinates of gauss points in current integration domain
# for k in range(len(x_G_domain)):
    
#     x_G_mn_k = 1.0/8.0*np.dot(np.array([(1+x_G_domain[k][0])*(1-x_G_domain[k][1])*(1-x_G_domain[k][2]), (1+x_G_domain[k][0])*(1+x_G_domain[k][1])*(1-x_G_domain[k][2]), \
#                             (1-x_G_domain[k][0])*(1+x_G_domain[k][1])*(1-x_G_domain[k][2]), (1-x_G_domain[k][0])*(1-x_G_domain[k][1])*(1-x_G_domain[k][2]), \
#                                 (1+x_G_domain[k][0])*(1-x_G_domain[k][1])*(1+x_G_domain[k][2]), (1+x_G_domain[k][0])*(1+x_G_domain[k][1])*(1+x_G_domain[k][2]), \
#                                     (1-x_G_domain[k][0])*(1+x_G_domain[k][1])*(1+x_G_domain[k][2]), (1-x_G_domain[k][0])*(1-x_G_domain[k][1])*(1+x_G_domain[k][2])],dtype=np.float64), np.transpose(x_ver_mn))
#     y_G_mn_k = 1.0/8.0*np.dot(np.array([(1+x_G_domain[k][0])*(1-x_G_domain[k][1])*(1-x_G_domain[k][2]), (1+x_G_domain[k][0])*(1+x_G_domain[k][1])*(1-x_G_domain[k][2]), \
#                             (1-x_G_domain[k][0])*(1+x_G_domain[k][1])*(1-x_G_domain[k][2]), (1-x_G_domain[k][0])*(1-x_G_domain[k][1])*(1-x_G_domain[k][2]), \
#                                 (1+x_G_domain[k][0])*(1-x_G_domain[k][1])*(1+x_G_domain[k][2]), (1+x_G_domain[k][0])*(1+x_G_domain[k][1])*(1+x_G_domain[k][2]), \
#                                     (1-x_G_domain[k][0])*(1+x_G_domain[k][1])*(1+x_G_domain[k][2]), (1-x_G_domain[k][0])*(1-x_G_domain[k][1])*(1+x_G_domain[k][2])],dtype=np.float64), np.transpose(y_ver_mn))
#     z_G_mn_k = 1.0/8.0*np.dot(np.array([(1+x_G_domain[k][0])*(1-x_G_domain[k][1])*(1-x_G_domain[k][2]), (1+x_G_domain[k][0])*(1+x_G_domain[k][1])*(1-x_G_domain[k][2]), \
#                             (1-x_G_domain[k][0])*(1+x_G_domain[k][1])*(1-x_G_domain[k][2]), (1-x_G_domain[k][0])*(1-x_G_domain[k][1])*(1-x_G_domain[k][2]), \
#                                 (1+x_G_domain[k][0])*(1-x_G_domain[k][1])*(1+x_G_domain[k][2]), (1+x_G_domain[k][0])*(1+x_G_domain[k][1])*(1+x_G_domain[k][2]), \
#                                     (1-x_G_domain[k][0])*(1+x_G_domain[k][1])*(1+x_G_domain[k][2]), (1-x_G_domain[k][0])*(1-x_G_domain[k][1])*(1+x_G_domain[k][2])],dtype=np.float64), np.transpose(z_ver_mn))
    
#     x_G.append([x_G_mn_k, y_G_mn_k, z_G_mn_k])

# print(x_G)
# comsol_left = [0.05425010319429718, 0.04933776441720655, 0.04439353469229062, 0.04149978241142322, 0.03944794287474251, \
#                0.03785780705961592, 0.03293687442739533, 0.02810516502993221, 0.02541045795105802, 0.02364482094207422, 0.02243690298876301]
# comsol_right = [0.33563547078597367, 0.3414636693350661, 0.3473297046019004, 0.35076297002004214, 0.3531973559109281, 0.35508395772546425, \
#                 0.3609223523738132, 0.3666548889471562, 0.36985199904069227, 0.371946822610617, 0.37337994560622545]

# meshfree_left = [0.054722640741973, 0.04926746885401708, 0.04438836849261093, 0.04150137528417908, 0.03944978166676898, \
#                  0.03785918565392357, 0.03293696849515196, 0.02810465496251630, 0.02540975529937669, 0.023644023292225407, 0.02243604863089717]
# meshfree_right = [0.33507480107031684, 0.34154703890348515, 0.34733580204413844, 0.3507610482236368, 0.35319514234599186, 0.3550822901578474, \
#                   0.36092220882083986, 0.3666554621648038, 0.36985280074841376, 0.371947737028326, 0.373380927304686]


# portion = [0.005, 0.01, 0.02, 0.03, 0.04,0.05,0.1,0.2,0.3,0.4,0.5]

# fig1 = plt.figure()
# plt.plot(portion, comsol_left, '-ro', label='COMSOL')
# plt.plot(portion, meshfree_left, '-bo', label='Meshfree')
# plt.legend()
# plt.grid()
# plt.xlabel('Portion of interface')
# plt.ylabel('Electolyte Potential')

# fig2 = plt.figure()
# plt.plot(portion, comsol_right, '-ro', label='COMSOL')
# plt.plot(portion, meshfree_right, '-bo', label='Meshfree')
# plt.legend()
# plt.grid()
# plt.xlabel('Portion of interface')
# plt.ylabel('Electode Potential')
# plt.show()

# a_spar = csc_matrix(a)


# c = np.ones(5)

# c11 = 0.06999256605498176
# c12 = 0.3169579724771405

# i_0 = 1.0e-3

# E_0 = 0.3
# v_app = 0.4

# c13 = i_0*np.exp(0.5*9.1148*(-c11+c12-E_0))
# print(c13)

# d11 = 0.06680066485543194
# d12 = 0.32074494092447564
# d13 = c13 = i_0*np.exp(0.5*9.1148*(-d11+d12-E_0))
# print(d13)
# exit()

# print(np.shape(a_spar))
# print(np.shape(b))

# aa = a_spar.T*a_spar

# print(aa.toarray())

# print(aa*b)

# phi_old_electrode= np.array(np.ones((10)))*0.3 # initial phi is 0.03
# phi_old_electrolyte = 0.01*np.array(np.ones((10)))

# a11 = np.concatenate((phi_old_electrolyte, phi_old_electrode))

# print(np.shape(a11))


# a22 = np.arange(1, 10)

# print(a22)

# print(type(a22), np.shape(a22))

# a33 = np.zeros((100, 100))

# print(a33[:, a22])


# aaa = csc_matrix(np.array([[1,0,0],[0,1,0],[0,0,1]]))
# from scipy.sparse.linalg import inv

# aaa1 = inv(aaa)

# aaa2 = np.zeros((3))

# aaa2[0] = 10
# aaa2[1] = 100
# aaa2[2] = 1000

# print(type(aaa1))

# print(type(inv(aaa1)*aaa2))

# print(aaa1[1,:2].toarray())

# i_HOR = np.exp(0.5*(aaa*aaa2))

# print(type(i_HOR))
# print(i_HOR)

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# b = np.array((10,20,30))

# print(np.shape(a))
# print(np.shape(b))

# print(a*b)
# print(np.dot(a,b))

# print(np.ones(2))


# aa = 1.234
# bb = [[aa,aa], [aa+1, aa+2]]
# print(bb)


# # potential_in_domain_save_electrolyte = np.loadtxt('potential_in_domain_electrolyte.txt')
# # potential_on_boundary_save_electrolyte = np.loadtxt('potential_on_boundary_electrolyte.txt')

# # potential_in_domain_save_electrode = np.loadtxt('potential_in_domain_electrode.txt')
# # potential_on_boundary_save_electrode = np.loadtxt('potential_on_boundary_electrode.txt')


# # fig1 = plt.figure()
# # plt.scatter(potential_in_domain_save_electrolyte[:, 0], potential_in_domain_save_electrolyte[:, 1], c=potential_in_domain_save_electrolyte[:, 2])
# # plt.scatter(potential_on_boundary_save_electrolyte[:, 0], potential_on_boundary_save_electrolyte[:, 1], c=potential_on_boundary_save_electrolyte[:, 2])

# # plt.colorbar()

# # fig2 = plt.figure()
# # plt.scatter(potential_in_domain_save_electrode[:, 0], potential_in_domain_save_electrode[:, 1], c=potential_in_domain_save_electrode[:, 2])
# # plt.scatter(potential_on_boundary_save_electrode[:, 0], potential_on_boundary_save_electrode[:, 1], c=potential_on_boundary_save_electrode[:, 2])

