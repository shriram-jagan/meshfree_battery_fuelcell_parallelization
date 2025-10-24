import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

a = np.array([1, 2, 3, 4, 5])

b = np.array([10, 20, 30, 40, 50])

print(a)
print(b)

a = np.array([11, 22, 33, 44, 55])
b[:] = a[:]

print(a)
print(b)

a = np.array([111, 222, 333, 444, 555])

print(a)
print(b)
