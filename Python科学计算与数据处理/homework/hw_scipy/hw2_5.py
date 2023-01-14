from scipy import integrate
import numpy as np


def f(x):
    return (np.cos(np.exp(x))) ** 2


def g(x, y):
    return 16 * x * y


result1 = integrate.quad(f, 0, 3)
result2 = integrate.dblquad(lambda x, y: 16 * x * y, 0, 0.5, 0, lambda x: (1 - 4 * x ** 2) ** 0.5)
print("解：", result1[0], "误差：", result1[1])
print("解：", result2[0], "误差：", result2[1])
