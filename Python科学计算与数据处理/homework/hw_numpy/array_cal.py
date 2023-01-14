import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

arr11 = 5 - np.arange(1, 13).reshape(4, 3)

sum_all = np.sum(arr11)
sum_row = np.sum(arr11, axis=1)
sum_col = np.sum(arr11, axis=0)
print("所有元素和：{}\n每一行的和：{}\n每一列的和：{}".format(sum_all, sum_row, sum_col))

cumsum_all = np.cumsum(arr11)
cumsum_row = np.cumsum(arr11, axis=1)
cumsum_col = np.cumsum(arr11, axis=0)
print("每个元素的累积和：{}\n每一行的累积和：{}\n每一列的累积和：{}".format(cumsum_all, cumsum_row, cumsum_col))

min_all = np.min(arr11)
max_col = np.max(arr11, axis=0)
print("所有元素的最小值：{}\n每一列的最大值：{}".format(min_all, max_col))

mean_all = np.mean(arr11)
mean_row = np.mean(arr11, axis=1)
print("所有元素的均值：{}\n每一行的均值：{}".format(mean_all, mean_row))

median_all = np.median(arr11)
median_col = np.median(arr11, axis=0)
print("所有元素的中位数：{}\n每一列的中位数：{}".format(median_all, median_col))

var_all = np.var(arr11)
std_row = np.std(arr11, axis=1)
print("所有元素的方差：{}\n每一行的标准差：{}".format(var_all, std_row))

a = np.array([1, 2, 3, 4, 5])
a = np.insert(a, [1, 2, 3, 4], 0)
a = np.insert(a, [2, 4, 6, 8], 0)

z = np.random.random((5, 5))
max_z = np.max(z)
min_z = np.min(z)
z_norm = (z - min_z) / (max_z - min_z)

z0 = np.random.random()
gap = np.abs(z - z0)
ans_idx = np.where(gap == np.min(gap))
ans = z[ans_idx]

w = np.array([[3, 6, -5], [1, -3, 2], [5, -1, 4]])
b = np.array([12, -2, 10])
solution = np.linalg.solve(w, b)


def f(x):
    y = (x - 1) * 5
    return np.sin(y ** 2) + (np.sin(y)) ** 2


xd = np.linspace(-1, 1, 1000)

n = 100
x1 = np.linspace(-1, 1, n)
x2 = Chebyshev.basis(n).roots()
c1 = Chebyshev.fit(x1, f(x1), n - 1, domain=[-1, 1])
c2 = Chebyshev.fit(x2, f(x2), n - 1, domain=[-1, 1])
print(abs(c1(xd) - f(xd)).max())
print(abs(c2(xd) - f(xd)).max())

data = pd.read_csv("height.csv")
A = data["A"]
B = data["B"]
sums = np.bincount(A, weights=B)[7:]
cnts = np.bincount(A)[7:]
print(sums / cnts)

money_list = []
money = 1000
for i in range(10000):
    if money < 0:
        break
    result = np.random.binomial(5, 0.5)
    if result < 3:
        money -= 8
    else:
        money += 8
    money_list.append(money)

plt.figure()
plt.plot(range(len(money_list)), money_list)
plt.show()
