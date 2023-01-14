# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def func(x, p):
    A, k, theta = p
    return A*np.sin(2*np.pi*k*x+theta)

def func_error(p, y, x):
    return np.sum((y - func(x, p))**2)

x = np.linspace(0, 2*np.pi, 100)
A, k, theta = 10, 0.34, np.pi/6
y0 = func(x, [A, k, theta])
np.random.seed(0)
y1 = y0 + 2 * np.random.randn(len(x))

result = optimize.basinhopping(func_error, (0., 0, 0),
    niter = 100,
    minimizer_kwargs={"method":"L-BFGS-B",
                        "args":(y1, x)})
print result.x

plt.plot(x, y1, "o", label=u"带噪声的实验数据")
plt.plot(x, y0, label=u"真实数据")
plt.plot(x, func(x, result.x), label=u"拟合数据")
plt.legend(loc="best");
plt.show()