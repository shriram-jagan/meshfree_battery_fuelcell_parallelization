import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from read_image import read_in_image

studied_physics = "fuel cell"
dimention = 3

file_name = 'micro_3d_connected.tif'
img_, unic_grain_id, num_pixels_xyz = read_in_image(file_name, studied_physics, dimention)

print(np.shape(img_))
exit()

ind = np.array([1,2,3])

print(img_[tuple(ind)])

exit()

# i_0 = 1.0e-3
# T = 1000.0+273.15

# E_0 = 1.0

# V_app = 1.4

# Fday = 9.6485e4     # Faraday constant
# R = 8.3145e0        # gas constant

# phi_old_node_electrolyte = 5

# print(i_0*np.exp(0.5*Fday/R/T*(-phi_old_node_electrolyte+V_app-E_0)))
# exit()

source1 = np.loadtxt('source0.txt')
source2 = np.loadtxt('source1.txt')

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
sc = ax.scatter(source1[:, 0], source1[:, 1],source1[:, 2], c=source1[:, 3])
plt.colorbar(sc, ax=ax)

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
sc = ax.scatter(source2[:, 0], source2[:, 1],source2[:, 2], c=source2[:, 3])
plt.colorbar(sc, ax=ax)


plt.show()
exit()

phi0 = np.loadtxt('phi_electrolyte_1.txt')
C0 = np.loadtxt('phi_electrode_1.txt')

phi1 = np.loadtxt('phi_electrolyte_2.txt')
C1 = np.loadtxt('phi_electrode_2.txt')

phi2 = np.loadtxt('phi_electrolyte_3.txt')
C2 = np.loadtxt('phi_electrode_3.txt')

phi3 = np.loadtxt('phi_electrolyte_4.txt')
C3 = np.loadtxt('phi_electrode_4.txt')

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
sc = ax.scatter(phi0[:, 0], phi0[:, 1],phi0[:, 2], c=phi0[:, 3])
plt.colorbar(sc, ax=ax)

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
sc = ax.scatter(phi1[:, 0], phi1[:, 1],phi1[:, 2], c=phi1[:, 3])
plt.colorbar(sc, ax=ax)

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
sc = ax.scatter(phi2[:, 0], phi2[:, 1],phi2[:, 2], c=phi2[:, 3])
plt.colorbar(sc, ax=ax)

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
sc = ax.scatter(phi3[:, 0], phi3[:, 1],phi3[:, 2], c=phi3[:, 3])
plt.colorbar(sc, ax=ax)


# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection='3d')
# sc = ax.scatter(C0[:, 0], C0[:, 1],C0[:, 2], c=C0[:, 3])
# plt.colorbar(sc, ax=ax)

# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection='3d')
# sc = ax.scatter(C1[:, 0], C1[:, 1],C1[:, 2], c=C1[:, 3])
# plt.colorbar(sc, ax=ax)

# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection='3d')
# sc = ax.scatter(C2[:, 0], C2[:, 1],C2[:, 2], c=C2[:, 3])
# plt.colorbar(sc, ax=ax)

# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection='3d')
# sc = ax.scatter(C3[:, 0], C3[:, 1],C3[:, 2], c=C3[:, 3])
# plt.colorbar(sc, ax=ax)

plt.show()
