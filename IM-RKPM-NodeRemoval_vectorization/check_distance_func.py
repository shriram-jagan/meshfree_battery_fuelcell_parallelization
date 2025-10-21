import numpy as np
import matplotlib.pyplot as plt

# bxbycxcy = np.loadtxt('BxByCxCy.txt')
# for i in range(np.shape(bxbycxcy)[0]):
#     plt.plot(bxbycxcy[i, 0], bxbycxcy[i, 1], 'k-o')
#     plt.plot(bxbycxcy[i, 2], bxbycxcy[i, 3], 'k-o')

# plt.show()

# exit()

distance_func_Xin_file = np.loadtxt('distance_func_in_domain.txt')
distance_func_dx_Xin_file = np.loadtxt('distance_func_dx_in_domain.txt')
distance_func_dy_Xin_file = np.loadtxt('distance_func_dy_in_domain.txt')
D_coor = np.loadtxt('points_D_coor.txt')

G_coor_Xin = distance_func_Xin_file[:, 1:3]
distance_func_Xin = distance_func_Xin_file[:, 3]
distance_func_dx_Xin = distance_func_dx_Xin_file[:, 3]
distance_func_dy_Xin = distance_func_dy_Xin_file[:, 3]



Kristen_file = np.loadtxt('Dist_dDistdx_dDistdy_123.txt', delimiter=",")
# D_file = np.loadtxt('GBcrd_refined_1000_filtered.txt')#, delimiter=",")


G_coor_Kristen = np.loadtxt('GIcrd_123.txt', delimiter=",")
# D_points_index = Kristen_file[:, 2]-1
distance_func_Kristen = Kristen_file[:, 0]
distance_func_dx_Kristen = Kristen_file[:, 1]
distance_func_dy_Kristen = Kristen_file[:, 2]

num_G = np.shape(G_coor_Xin)[0]
num_G_K = np.shape(G_coor_Kristen)[0]
print(num_G, num_G_K)

allign_index = []

distance_func_Kristen_alligned = []
distance_func_dx_Kristen_alligned = []
distance_func_dy_Kristen_alligned = []
# D_point_index_alligned = []

for i in range(num_G):
    for j in range(num_G):
        if abs(G_coor_Xin[i, 0] - G_coor_Kristen[j, 0]) < 1e-10 and abs(G_coor_Xin[i, 1] - G_coor_Kristen[j, 1]) < 1e-10:
            # D_point_index_alligned.append(D_points_index[j])
            allign_index.append(j)
            distance_func_Kristen_alligned.append(distance_func_Kristen[j])
            distance_func_dx_Kristen_alligned.append(distance_func_dx_Kristen[j])
            distance_func_dy_Kristen_alligned.append(distance_func_dy_Kristen[j])

print(len(allign_index))
# exit()

G_coor_kris_allgined = G_coor_Kristen[allign_index][:]

print(np.max(G_coor_kris_allgined-G_coor_Xin))

bxbycxcy = np.loadtxt('bxbycxcy.txt')

max_diff_index = np.argmax(abs(np.asarray(distance_func_dy_Xin-distance_func_dy_Kristen_alligned)))

print('max diff index', max_diff_index)

print('x_G_xin',G_coor_Xin[max_diff_index, :])
# print('x_D_xin', D_coor[max_diff_index, :])
# print('x_D_Kristen', D_file[int(D_point_index_alligned[max_diff_index]), :])
print('distance_func_xin', distance_func_Xin[max_diff_index])
print('distance_func_Kristen', distance_func_Kristen_alligned[max_diff_index])
print('distance_func_dx_xin', distance_func_dx_Xin[max_diff_index])
print('distance_func_dx_Kristen', distance_func_dx_Kristen_alligned[max_diff_index])
print('distance_func_dy_xin', distance_func_dy_Xin[max_diff_index])
print('distance_func_dy_Kristen', distance_func_dy_Kristen_alligned[max_diff_index])
# print('distance_func_xin', distance_func_Xin[max_diff_index])
# print('distance_func_Kristen', distance_func_Kristen_alligned[max_diff_index])

fig5 = plt.figure()
plt.plot(bxbycxcy[:,0], bxbycxcy[:,1], 'bo')
plt.plot(bxbycxcy[:,2], bxbycxcy[:,3], 'bo')

plt.plot(-8.42264973e-07,  4.55773503e-06, 'ro')

plt.scatter(G_coor_Xin[:,0], G_coor_Xin[:,1],c=distance_func_Xin-distance_func_Kristen_alligned)

# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dy_Xin-distance_func_dy_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# plt.savefig('grad_x_shape_func_not_on_interface.png')
plt.colorbar()
# plt.show()

fig6 = plt.figure()
plt.plot(bxbycxcy[:,0], bxbycxcy[:,1], 'bo')
plt.plot(bxbycxcy[:,2], bxbycxcy[:,3], 'bo')

plt.scatter(G_coor_Xin[:,0], G_coor_Xin[:,1],c=distance_func_dy_Xin-distance_func_dy_Kristen_alligned)

# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dy_Xin-distance_func_dy_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# plt.savefig('grad_x_shape_func_not_on_interface.png')
plt.colorbar()
# plt.show()

fig6 = plt.figure()
plt.plot(bxbycxcy[:,0], bxbycxcy[:,1], 'bo')
plt.plot(bxbycxcy[:,2], bxbycxcy[:,3], 'bo')

plt.scatter(G_coor_Xin[:,0], G_coor_Xin[:,1],c=distance_func_dx_Xin-distance_func_dx_Kristen_alligned)

# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dy_Xin-distance_func_dy_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# plt.savefig('grad_x_shape_func_not_on_interface.png')
plt.colorbar()
plt.show()

# fig4 = plt.figure()
# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_Xin-distance_func_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# # ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# # plt.savefig('grad_x_shape_func_not_on_interface.png')
# plt.show()

# fig5 = plt.figure()
# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dx_Xin, color='white', edgecolors='grey', alpha=0.5)
# # ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# # plt.savefig('grad_x_shape_func_not_on_interface.png')
# plt.show()

# fig7 = plt.figure()
# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dx_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# # ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# # plt.savefig('grad_x_shape_func_not_on_interface.png')
# plt.show()

# # fig7 = plt.figure()
# # ax = plt.axes(projection ='3d')

# # ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dx_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# # # ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# # # plt.savefig('grad_x_shape_func_not_on_interface.png')
# # plt.show()

# fig8 = plt.figure()
# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], (np.array(distance_func_dx_Kristen_alligned)-np.array(distance_func_dx_Xin))/np.array(distance_func_dx_Kristen_alligned), color='white', edgecolors='grey', alpha=0.5)
# # ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# # plt.savefig('grad_x_shape_func_not_on_interface.png')
# plt.show()

# fig6 = plt.figure()
# ax = plt.axes(projection ='3d')

# ax.plot_trisurf(G_coor_Xin[:,0], G_coor_Xin[:,1], distance_func_dy_Xin-distance_func_dy_Kristen_alligned, color='white', edgecolors='grey', alpha=0.5)
# # ax.plot_trisurf(G_coor_Kristen[:,0], G_coor_Kristen[:,1], distance_func_Kristen_alligned, color='white', edgecolors='blue', alpha=0.5)
# # plt.savefig('grad_x_shape_func_not_on_interface.png')
# plt.show()


