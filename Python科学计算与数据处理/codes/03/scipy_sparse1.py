# -*- coding: utf-8 -*-
import numpy as np 
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import scipy.sparse 
import time
N = 3000
# 创建随机稀疏矩阵 
m = scipy.sparse.rand(N, N) 
# 创建包含相同数据的数组 
a = m.toarray()
print('The numpy array data size: ' + str(a.nbytes) + ' bytes')
print('The sparse matrix data size: ' + str(m.data.nbytes) + ' bytes') 
# 数组求特征值 
t0 = time.time() 
res1 = eigh(a)
dt = str(np.round(time.time() - t0, 3)) + ' seconds' 
print('Non-sparse operation takes ' + dt) 
# 稀疏长阵求特征值 
t0 = time.time() 
res2 = eigsh(m) 
dt = str(np.round(time.time() - t0, 3)) + ' seconds' 
print('Sparse operation takes ' + dt)