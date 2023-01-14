import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x = np.array([-1, 0, 2.0, 1.0])
y = np.array([1.0, 0.3, -0.5, 0.8])
xd = np.linspace(-3, 4, 100)
f_multiquadric = interpolate.Rbf(x, y, function='multiquadric')
f_gaussian = interpolate.Rbf(x, y, function='gaussian')
f_linear = interpolate.Rbf(x, y, function='linear')

yd_multiquadric = f_multiquadric(xd)
yd_gaussian = f_gaussian(xd)
yd_linear = f_linear(xd)

plt.figure()
plt.plot(xd, yd_multiquadric, label="multiquadric")
plt.plot(xd, yd_gaussian, label="gaussian")
plt.plot(xd, yd_linear, label="linear")
plt.scatter(x, y)
plt.legend()
plt.show()
