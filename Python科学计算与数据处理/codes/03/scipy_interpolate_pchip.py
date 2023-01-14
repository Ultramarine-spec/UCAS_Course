# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
x = [0, 1, 2, 3, 4, 5]
y = [1, 2, 1.5, 8, 6, 2.5]
xs = np.linspace(x[0], x[-1], 100)
curve = interpolate.pchip(x, y)
ys = curve(xs)
dcurve = curve.derivative()
dys = dcurve(xs)
plt.plot(xs, ys, label=u"pchip")
plt.plot(xs, dys, label=u"First derivative")
plt.plot(x, y, "o")
plt.legend(loc="best") 
plt.grid()
plt.margins(0.1, 0.9)
plt.show()