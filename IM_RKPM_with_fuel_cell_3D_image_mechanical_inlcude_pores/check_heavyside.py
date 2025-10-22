import matplotlib.pyplot as plt
import numpy as np
from numba import jit

x_G = np.loadtxt("x_G.txt")
heavy_xin = np.loadtxt("heavyside_124.txt")
heavy_px_xin = np.loadtxt("heavyside_px_124.txt")
heavy_py_xin = np.loadtxt("heavyside_py_124.txt")

kristen_data = np.loadtxt("Heavi_dHeavidx_dHeavidyIMRK_eps_123.txt", delimiter=",")

x_G_kristen = np.loadtxt("GIcrd_123.txt", delimiter=",")


@jit
def allign_G(x_G, x_G_kristen, kristen_data):

    alligned_x_G = []
    alligned_heavy = []
    alligned_heavy_px = []
    alligned_heavy_py = []

    for i in range(np.shape(x_G)[0]):
        for j in range(np.shape(x_G)[0]):

            if (
                abs(x_G[i, 0] - x_G_kristen[j, 0]) < 1.0e-13
                and abs(x_G[i, 1] - x_G_kristen[j, 1]) < 1.0e-13
            ):
                alligned_x_G.append([kristen_data[j, 0], kristen_data[j, 1]])
                alligned_heavy.append(kristen_data[j, 0])
                alligned_heavy_px.append(kristen_data[j, 1])
                alligned_heavy_py.append(kristen_data[j, 2])

    return alligned_x_G, alligned_heavy, alligned_heavy_px, alligned_heavy_py


alligned_x_G, alligned_heavy, alligned_heavy_px, alligned_heavy_py = allign_G(
    x_G, x_G_kristen, kristen_data
)

print(np.max(abs(alligned_x_G - x_G)))

print(np.shape(alligned_x_G), np.shape(alligned_heavy_px), np.shape(alligned_heavy_py))

plt.figure()
plt.scatter(x_G[:, 0], x_G[:, 1], c=abs(alligned_heavy - heavy_xin))
plt.colorbar()
plt.title("difference of h")

plt.figure()
plt.scatter(x_G[:, 0], x_G[:, 1], c=abs(alligned_heavy_px - heavy_px_xin))
plt.colorbar()
plt.title("difference of h_x")

plt.figure()
plt.scatter(x_G[:, 0], x_G[:, 1], c=abs(alligned_heavy_py - heavy_py_xin))
plt.colorbar()
plt.title("difference of h_y")
plt.show()
