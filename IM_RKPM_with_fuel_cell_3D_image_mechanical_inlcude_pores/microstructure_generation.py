import tifffile
import numpy as np
from scipy.ndimage import label, generate_binary_structure

# simple 2d with 3-phase
M_3d_simple = np.zeros((200,100)).astype(np.uint8)
M_3d_simple[:100,:] = 2
M_3d_simple[100:,:50] = 1
tifffile.imwrite('M_2d_3phases_simple.tif', M_3d_simple, compression='lzw')
exit()

# simple 3d with 3-phase
M_3d_simple = np.zeros((20,40,20)).astype(np.uint8)
M_3d_simple[:,:20,:] = 2
M_3d_simple[:,20:,:10] = 1
tifffile.imwrite('M_3d_3phases_simple.tif', M_3d_simple, compression='lzw')

exit()





M = tifffile.imread('Synthetic_25_20_55_cut_8_bit.tif')

M_extracted = M[:80, :80,:80]
M_extracted_new = np.zeros((20,20,20))

for i in range(20):
    for j in range(20):
        for k in range(20):
            id_of_initial = M_extracted[4*i:4*i+4, 4*j:4*j+4, 4*k:4*k+4]
            if np.average(id_of_initial)<0.5:
                M_extracted_new[i,j,k] = 0
            if np.average(id_of_initial)>=0.5 and np.average(id_of_initial)<1.5:
                M_extracted_new[i,j,k] = 1
            if np.average(id_of_initial)>=1.5 and np.average(id_of_initial)<=2:
                M_extracted_new[i,j,k] = 2

np.flip(M_extracted_new, axis=1)

M_extracted_new = M_extracted_new.astype(np.uint8)

M_all = (np.ones((20, 40,20))*2).astype(np.uint8)

M_all[:, 20:,:] = M_extracted_new

M_2_connect = M_all.copy()
M_2_connect[np.where(M_all==1)] = 0
M_1_connect = M_all.copy()
M_1_connect[np.where(M_all==2)] = 0

structure = generate_binary_structure(3, 1)  # 26-connectivity

# Label connected regions
labeled_2, num_2 = label(M_2_connect, structure)
tifffile.imwrite('M_2_discontinuous.tif', labeled_2, compression='lzw')
labeled_2[np.where(labeled_2>1)]=0

# Label connected regions
labeled_1, num_1 = label(M_1_connect, structure)
tifffile.imwrite('M_1_discontinuous.tif', labeled_1, compression='lzw')
labeled_1[np.where(labeled_1>1)]=0

M_connected = np.zeros((20,40,20)).astype(np.uint8)
M_connected[np.where(labeled_1==1)] = 1
M_connected[np.where(labeled_2==1)] = 2

tifffile.imwrite('M_all.tif', M_all, compression='lzw')
tifffile.imwrite('M_1_connect.tif', labeled_1, compression='lzw')
tifffile.imwrite('M_2_connect.tif', labeled_2, compression='lzw')
tifffile.imwrite('M_connect.tif', M_connected, compression='lzw')


print(np.unique(labeled_1))
print(num_1)
exit()

# exit()

tifffile.imwrite('M_extracted_new.tif', M_extracted_new, compression='lzw')
tifffile.imwrite('M_all.tif', M_all, compression='lzw')