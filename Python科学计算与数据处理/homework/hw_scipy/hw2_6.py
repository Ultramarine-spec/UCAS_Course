import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


def F(y, t, m, b, k, f):
    theta, omega = y
    dydt = [omega, (f - b * omega - k * theta) / m]
    return dydt


m = 1
b = 0.2
k = 0.5
f = 1
y0 = [-1, 0]
t = np.arange(0, 50, 0.02)
result = integrate.odeint(F, y0, t, args=(m, b, k, f))

plt.plot(t, result[:, 0], 'b', label='theta(t)')
plt.plot(t, result[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
