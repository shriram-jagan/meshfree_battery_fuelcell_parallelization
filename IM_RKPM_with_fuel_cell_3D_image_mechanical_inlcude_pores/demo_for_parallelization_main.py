import time

import matplotlib.pyplot as plt
import numpy as np
from demo_for_parallelization_call import func_1, func_2
from numba import jit
from numpy import sign
from numpy.linalg import eig, norm
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from tqdm import tqdm

a1 = np.loadtxt("a1.txt")
a2 = np.loadtxt("a2.txt")
a1_num = np.shape(a1)[0]
a2_num = np.shape(a2)[0]


a3 = np.loadtxt("a3.txt")
a4 = np.loadtxt("a4.txt")

c = 2

h = np.zeros(a2_num)

for i in range(a2_num):
    di = ((a2[i, 0] - a2[:, 0]) ** 2 + (a2[i, 1] - a2[:, 1]) ** 2) ** 0.5

    index_four_smallest = sorted(range(len(di)), key=lambda sub: di[sub])[
        :5
    ]  # get the index of the four smallest index, the first one is always zero, so 5 here

    h[i] = di[index_four_smallest][
        di[index_four_smallest].tolist().index(max(di[index_four_smallest]))
    ]

    a = c * h  # shape: (a2_num,)

M1 = np.array([np.zeros((3, 3)) for _ in range(a1_num)])
M2 = np.array([np.zeros((3, 3)) for _ in range(a1_num)])  # partial M partial x
M3 = np.array([np.zeros((3, 3)) for _ in range(a1_num)])  # partial M partial y


a5 = np.loadtxt("a5.txt")

a6 = np.loadtxt("a6.txt")

a6_num = np.shape(a6)[0]


n1, n2, n3, n4, n5, M1, M2, M3 = func_1(a1, a3, a2, a4, a, M1, M2, M3, a6_num, a5, a6)


n1_num = np.shape(np.array(n1))[0]

H0 = np.array([1, 0, 0], dtype=np.float64)

s1, s2, s3 = func_2(a1, a2, n1_num, H0, M1, M2, M3, n3, n4, n5, n1, n2)
