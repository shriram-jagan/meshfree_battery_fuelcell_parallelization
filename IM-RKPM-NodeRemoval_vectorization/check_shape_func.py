import numpy as np
from numba import jit
import matplotlib.pyplot as plt

shape_func = np.loadtxt('shape_func_124.txt')
grad_shape_func_x = np.loadtxt('grad_shape_func_x_124.txt')
grad_shape_func_y = np.loadtxt('grad_shape_func_y_124.txt')

print('shape function', np.max(shape_func), np.min(shape_func))
print('grad shape function grad x', np.max(grad_shape_func_x), np.min(grad_shape_func_x))
print('grad shape function grad y', np.max(grad_shape_func_y), np.min(grad_shape_func_y))

x_G = np.loadtxt('x_G.txt')
x_nodes = np.loadtxt('x_nodes.txt')

x_G_kristen = np.loadtxt('GIcrd_123.txt', delimiter=',')
x_nodes_kristen = np.loadtxt('crd_124.txt', delimiter=',')

shape_func_kristen = np.loadtxt('shpIMRK_123.txt', delimiter=',')
grad_shape_func_x_kristen = np.loadtxt('dshpxIMRK_123.txt', delimiter=',')
grad_shape_func_y_kristen = np.loadtxt('dshpyIMRK_123.txt', delimiter=',')

print('G_shape', np.shape(x_G), np.shape(x_G_kristen))
print('nodes_shape', np.shape(x_nodes), np.shape(x_nodes_kristen))
print('shapefunc_shape', np.shape(shape_func), np.shape(shape_func_kristen))
print('grad_shapefunc_x_shape', np.shape(grad_shape_func_x), np.shape(grad_shape_func_x_kristen))
print('grad_shapefunc_y_shape', np.shape(grad_shape_func_y), np.shape(grad_shape_func_y_kristen))

num_G = np.shape(x_G)[0]
num_nodes = np.shape(x_nodes)[0]

@jit

def allign_G_nodes(num_G, num_nodes, x_G, x_nodes, x_G_kristen, x_nodes_kristen, shape_func, shape_func_kristen, grad_shape_func_x, grad_shape_func_x_kristen, grad_shape_func_y, grad_shape_func_y_kristen):
    alligned_shape_func = np.zeros((num_G, num_nodes))
    inter_alligned_shape_func = np.zeros((num_G, num_nodes))
    alligned_grad_shape_func_x = np.zeros((num_G, num_nodes))
    inter_alligned_grad_shape_func_x = np.zeros((num_G, num_nodes))
    alligned_grad_shape_func_y = np.zeros((num_G, num_nodes))
    inter_alligned_grad_shape_func_y = np.zeros((num_G, num_nodes))
    # non_zero_shape_func_index_i_kristen = []
    # non_zero_shape_func_index_j_kristen = []
    # non_zero_shape_func_index_i_xin = []
    # non_zero_shape_func_index_j_xin = []
    mismatched_index_shape = []
    mismatched_index_grad_shape_x = []
    mismatched_index_grad_shape_y = []
    alligned_x_G_kristen = []
    alligned_x_nodes_kristen = []

    # non_zero_grad_shape_x_Xin = 0
    # non_zero_grad_shape_y_Xin = 0

    # non_zero_grad_shape_x_Kristen = 0
    # non_zero_grad_shape_y_Kristen = 0

    non_zero_shape_Xin = np.shape(np.where(shape_func!=0)[0])
    non_zero_shape_Kristen = np.shape(np.where(shape_func_kristen!=0)[0])

    non_zero_grad_shape_x_Kristen = np.shape(np.where(grad_shape_func_x_kristen!=0)[0])
    non_zero_grad_shape_y_Kristen = np.shape(np.where(grad_shape_func_y_kristen!=0)[0])
    non_zero_grad_shape_x_Xin = np.shape(np.where(grad_shape_func_x!=0)[0])
    non_zero_grad_shape_y_Xin = np.shape(np.where(grad_shape_func_y!=0)[0])

    print('num of nonzero shap func',non_zero_shape_Xin, non_zero_shape_Kristen)
    print('num of nonzero grad shap func x',non_zero_grad_shape_x_Xin, non_zero_grad_shape_x_Kristen)
    print('num of nonzero grad shap func y',non_zero_grad_shape_y_Xin, non_zero_grad_shape_y_Kristen)

    for i in range(num_G):
        for j in range(num_G):
            if abs(x_G[i, 0] - x_G_kristen[j, 0]) < 1e-13 and abs(x_G[i, 1] - x_G_kristen[j, 1]) < 1e-13:
                alligned_x_G_kristen.append([x_G[i, 0], x_G[i, 1]])
                
                alligned_shape_func[i, :] = shape_func_kristen[j, :]
                inter_alligned_shape_func[i, :] = shape_func_kristen[j, :]
                alligned_grad_shape_func_x[i, :] = grad_shape_func_x_kristen[j, :]
                inter_alligned_grad_shape_func_x[i, :] = grad_shape_func_x_kristen[j, :]
                alligned_grad_shape_func_y[i, :] = grad_shape_func_y_kristen[j, :]
                inter_alligned_grad_shape_func_y[i, :] = grad_shape_func_y_kristen[j, :]
    for m in range(num_nodes):
        for n in range(num_nodes):
            if abs(x_nodes[m, 0] - x_nodes_kristen[n, 0]) < 1e-13 and abs(x_nodes[m, 1] - x_nodes_kristen[n, 1]) < 1e-13:
                alligned_x_nodes_kristen.append([x_nodes[m, 0], x_nodes[m, 1]])
           
                alligned_shape_func[:, m] = inter_alligned_shape_func[:, n]
                alligned_grad_shape_func_x[:, m] = inter_alligned_grad_shape_func_x[:, n]
                alligned_grad_shape_func_y[:, m] = inter_alligned_grad_shape_func_y[:, n]
    
    aa_shape = 0
    aa_grad_shape_x = 0
    aa_grad_shape_y = 0
    
    # mismatch_shape = []
    mismatch_grad_x = []
    mismatch_grad_y = []

    for i in range(num_G):
        for j in range(num_nodes):
            if abs(alligned_shape_func[i][j]-shape_func[i][j])>1.0e-5:
                aa_shape += 1
                mismatched_index_shape.append([i,j])
                # mismatch_shape
            if abs(alligned_grad_shape_func_x[i][j]-grad_shape_func_x[i][j])> 1.0e-5:
                aa_grad_shape_x += 1
                mismatched_index_grad_shape_x.append([i,j])
                mismatch_grad_x.append(abs(alligned_grad_shape_func_x[i][j]-grad_shape_func_x[i][j]))
            if abs(alligned_grad_shape_func_y[i][j]-grad_shape_func_y[i][j])> 1.0e-5:
                aa_grad_shape_y += 1
                mismatched_index_grad_shape_y.append([i,j])
                mismatch_grad_y.append(abs(alligned_grad_shape_func_y[i][j]-grad_shape_func_y[i][j]))
                # break
        
    #     if shape_func_kristen[i,j] != 0:
    #         non_zero_shape_func_index_i_kristen.append(i)
    #         non_zero_shape_func_index_j_kristen.append(j)
    #     if abs(shape_func[i,j]) > 1.0e-12:
    #         non_zero_shape_func_index_i_xin.append(i)
    #         non_zero_shape_func_index_j_xin.append(j)
    # print(len(non_zero_shape_func_index_i_kristen), len(non_zero_shape_func_index_j_kristen), len(non_zero_shape_func_index_i_xin), len(non_zero_shape_func_index_j_xin))
    # for m in range(num_G):
    #     for n in range(num_nodes):
    #         if abs(x_G[m, 0] - x_G_kristen[i, 0]) < 1e-13 and abs(x_G[m, 1] - x_G_kristen[i, 1]) < 1e-13 and abs(x_nodes[n, 0] - x_nodes_kristen[j, 0]) < 1e-13 and abs(x_nodes[n, 1] - x_nodes_kristen[j, 1]) < 1e-13:
    #             alligned_shape_func[i][j] = shape_func[m][n]
    #             alligned_grad_shape_func_x[i][j] = grad_shape_func_x[m][n]
    #             alligned_grad_shape_func_y[i][j] = grad_shape_func_y[m][n]
            
    
    # print(shape_func_kristen[188,46])
    # print(shape_func_kristen[4, 163])
    return alligned_x_nodes_kristen, alligned_x_G_kristen, mismatch_grad_x, mismatch_grad_y, aa_shape, aa_grad_shape_x, aa_grad_shape_y, mismatched_index_shape, mismatched_index_grad_shape_x, mismatched_index_grad_shape_y, alligned_shape_func, alligned_grad_shape_func_x, alligned_grad_shape_func_y

# alligned_shape_func, alligned_grad_shape_func_x, alligned_grad_shape_func_y = 
alligned_x_nodes_kristen, alligned_x_G_kristen,mismatch_grad_x, mismatch_grad_y, aa_shape, aa_grad_shape_x, aa_grad_shape_y, mismatched_index_shape, mismatched_index_grad_shape_x, mismatched_index_grad_shape_y, alligned_shape_func, alligned_grad_shape_func_x, alligned_grad_shape_func_y =  allign_G_nodes(num_G, num_nodes, x_G, x_nodes, x_G_kristen, x_nodes_kristen, shape_func, shape_func_kristen, grad_shape_func_x, grad_shape_func_x_kristen, grad_shape_func_y, grad_shape_func_y_kristen)
# np.savetxt('mismatched_index_shape.txt', np.array(mismatched_index_shape))
# np.savetxt('mismatched_index_shape_x.txt', np.array(mismatched_index_grad_shape_x))
# np.savetxt('mismatched_index_shape_y.txt', np.array(mismatched_index_grad_shape_y))
# np.savetxt('alligned_shape_func.txt', alligned_shape_func)
# np.savetxt('alligned_shape_func_x.txt', alligned_grad_shape_func_x)
# np.savetxt('alligned_shape_func_y.txt', alligned_grad_shape_func_y)

# print(np.shape(mismatched_index_shape), np.shape(mismatched_index_grad_shape_x), np.shape(mismatched_index_grad_shape_y))
# print(aa_shape, aa_grad_shape_x, aa_grad_shape_y)
print(np.linalg.norm(alligned_shape_func-shape_func, 2))
print(np.linalg.norm(alligned_grad_shape_func_x-grad_shape_func_x, 2))
print(np.linalg.norm(alligned_grad_shape_func_y-grad_shape_func_y, 2))
# print(shape_func[104, 56])
# print(alligned_shape_func[104, 56])
# print(shape_func_kristen[4808, 2753])

# print(type(np.linalg.norm(x_nodes-np.asarray(alligned_x_nodes_kristen),2,axis=1)), np.shape(np.linalg.norm(x_nodes-np.asarray(alligned_x_nodes_kristen),2,axis=1)))

# print(x_nodes[np.array(mismatched_index_grad_shape_x)[:,1], :])

# print(np.max(np.linalg.norm(x_G-np.asarray(alligned_x_G_kristen),2,axis=1)))
# print(np.max(np.linalg.norm(x_nodes-np.asarray(alligned_x_nodes_kristen),2,axis=1)))

# plt.figure()
# # plt.scatter(x_G[:, 0], x_G[:, 1], c=np.linalg.norm(x_G-np.asarray(alligned_x_G_kristen),2,axis=1))
# plt.scatter(x_G[np.array(mismatched_index_shape)[:,0], 0], x_G[np.array(mismatched_index_shape)[:,0], 1], c=mismatch_)
# plt.colorbar()
# plt.title('log(difference of x_G)')


plt.figure()
# plt.scatter(x_G[:, 0], x_G[:, 1], c=np.linalg.norm(x_G-np.asarray(alligned_x_G_kristen),2,axis=1))
plt.scatter(x_G[np.array(mismatched_index_grad_shape_x)[:,0], 0], x_G[np.array(mismatched_index_grad_shape_x)[:,0], 1], c=mismatch_grad_x)
plt.colorbar()
plt.title('log(difference of x_G)')
# plt.show()
plt.figure()
# plt.scatter(x_nodes[:, 0], x_nodes[:, 1], c=np.linalg.norm(x_nodes-np.asarray(alligned_x_nodes_kristen),2,axis=1))

plt.scatter(x_nodes[np.array(mismatched_index_grad_shape_y)[:,1], 0], x_nodes[np.array(mismatched_index_grad_shape_y)[:,1], 1], c = mismatch_grad_y)
plt.colorbar()
plt.title('log(diffrence of x_nodes)')
plt.show()