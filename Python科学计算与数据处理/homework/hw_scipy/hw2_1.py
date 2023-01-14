import math
import time
import numpy as np
from scipy.optimize import fsolve


def f(x):
    a, r = x
    d = 140
    l = 156
    return [math.cos(a) - 1 + d * d / (2 * r * r), l - a * r]


def derivative_f(x):
    a, r = x
    d = 140
    l = 156
    return [[-math.sin(a), -(d * d) / (r * r * r)], [-r, -a]]


x0 = np.array([1, 1])
result1 = fsolve(f, x0)
result2 = fsolve(f, x0, fprime=derivative_f)
print("不导入雅克比矩阵得到的解：{}".format(result1))
print("导入雅克比矩阵得到的解：{}".format(result2))

start = time.process_time()
for _ in range(100000):
    fsolve(f, x0)
print("不导入雅克比矩阵平均运算时间：{}".format((time.process_time() - start) / 100000))

start = time.process_time()
for _ in range(100000):
    fsolve(f, x0, fprime=derivative_f)
print("导入雅克比矩阵平均运算时间：{}".format((time.process_time() - start) / 100000))
