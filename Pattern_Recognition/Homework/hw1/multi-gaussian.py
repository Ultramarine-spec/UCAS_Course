import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mean_1 = [1, 0]
mean_2 = [-1, 0]
cov = [[1, 0],
       [0, 1]]
pw1 = 0.5
pw2 = 0.5

error_list = []
for n in range(100, 1000, 100):
    x1, y1 = np.random.multivariate_normal(mean_1, cov, n // 2).T
    point_1 = list(zip(x1, y1))

    x2, y2 = np.random.multivariate_normal(mean_2, cov, n // 2).T
    point_2 = list(zip(x2, y2))

    point_1_error = [p for p in point_1 if p[0] < 0]
    point_2_error = [p for p in point_2 if p[0] > 0]

    error = (len(point_1_error) + len(point_2_error)) / n
    error_list.append(error)

plt.figure()
plt.plot(range(100, 1000, 100), error_list, label="Error")
plt.xlabel("Number of sampling")
plt.ylabel("Error")
plt.legend()
plt.show()
