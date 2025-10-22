import numpy as np
from scipy.sparse import csc_matrix

# arr_small = np.array([[1, 2, 3],
#                       [4, 5, 6]])            # shape (2, 3)
arr_large = np.array(
    [[9, 9, 0], [1, 0, 3], [0, 8, 9], [4, 0, 0], [0, 0, 0]]
)  # shape (5, 3)

csc = csc_matrix(arr_large)

array = np.array([1, 2, 3, 4, 5])

csc.data *= array[csc.indices]

print(csc.toarray())

# # Expand both arrays for broadcasting
# # arr_small[:, None, :] → shape (2, 1, 3)
# # arr_large[None, :, :] → shape (1, 5, 3)
# matches = np.all(arr_small[:, None, :] == arr_large[None, :, :], axis=2)
# print(matches)
# # For each row in arr_small, get index in arr_large where match is True
# indices = np.argmax(matches, axis=1)  # gets first match per row
# print(indices)
# print(np.shape(arr_large))
# print(arr_large[indices])
# exit()
# Optionally, ensure that a match was found:
# mask = matches.any(axis=1)
# indices[~mask] = -1  # mark unmatched rows with -1

# print(indices)

# a = np.zeros((10,10,10))

# adjacent_pixel_index = np.array([[2,3,3],[3,2,3],[2,22,3],[2,3,2],[-1,2,2],[2,2,2], [3,3,2]]) # 7 adjacent pixels
# print(adjacent_pixel_index[np.all(adjacent_pixel_index>=0,axis=1)])
# exit()
# print(tuple(adjacent_pixel_index.T))
# a[tuple(adjacent_pixel_index.T)] = 100
# a[2,3,3] = 10
# a[2,3,2] = 20
# cc = np.unique(a[tuple(adjacent_pixel_index.T)])

# if 20 in cc and 10 in cc:
#     print('yes')

# print(a[2,3,3])
# print(a[3,2,3])
# print(a[2,2,3])
# print(a[2,3,2])
# print(a[3,2,2])
# print(a[2,2,2])
# print(a[3,3,2])
