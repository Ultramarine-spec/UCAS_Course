import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

p_x = []
p_y = []


def f(x):
    y = x ** 2 + 10 * np.sin(x)
    p_x.append(x)
    p_y.append(y)
    return y


def derivative_f(x):
    return 2 * x + 10 * np.cos(x)


xd = np.linspace(-10, 10, 100)
x0 = np.array(10)
result1 = opt.fmin_bfgs(f, x0, derivative_f)
plt.figure()
plt.scatter(p_x, p_y, c=range(len(p_x)))
for i in range(len(p_x) - 1):
    plt.quiver(p_x[i], p_y[i], p_x[i + 1] - p_x[i], p_y[i + 1] - p_y[i], angles='xy', scale=1, scale_units='xy',
               width=0.004)
plt.plot(xd, f(xd), label="fmin_bfgs")
plt.legend()
plt.show()

p_x = []
p_y = []
result2 = opt.fminbound(f, -10, 10)
plt.figure()
plt.scatter(p_x, p_y, c=range(len(p_x)))
for i in range(len(p_x) - 1):
    plt.quiver(p_x[i], p_y[i], p_x[i + 1] - p_x[i], p_y[i + 1] - p_y[i], angles='xy', scale=1, scale_units='xy',
               width=0.004)
plt.plot(xd, f(xd), label="fminbound")
plt.legend()
plt.show()

p_x = []
p_y = []
result3 = opt.brent(f)
plt.figure()
plt.scatter(p_x, p_y, c=range(len(p_x)))
for i in range(len(p_x) - 1):
    plt.quiver(p_x[i], p_y[i], p_x[i + 1] - p_x[i], p_y[i + 1] - p_y[i], angles='xy', scale=1, scale_units='xy',
               width=0.004)
plt.plot(xd, f(xd), label="brent")
plt.legend()
plt.show()
