# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

rand = np.random.RandomState(42)
X = rand.rand(10, 2)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], s=100)

dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=2)

nearest = np.argsort(dist_sq, axis=1) 
print(nearest)

K =2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

plt.scatter(X[:, 0], X[:, 1], s=100)
# 将每个点与它的两个最近邻连接
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]: # 画一条从X[i]到X[j]的线段
# 用zip方法实现：
        plt.plot(*zip(X[j], X[i]), color='black')
plt.show()
