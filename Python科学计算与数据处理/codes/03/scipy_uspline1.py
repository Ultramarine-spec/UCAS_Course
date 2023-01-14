# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
x1 = np.linspace(0, 10, 20)
y1 = np.sin(x1)
sx1 = np.linspace(0, 12, 100)
sy1 = interpolate.UnivariateSpline(x1, y1, s=0)(sx1)
#外推运算使得输入数据x的值没有大于10的点但是能计算出在10到12之间的数值
x2 = np.linspace(0, 20, 200)
y2 = np.sin(x2) + np.random.standard_normal(len(x2))*0.2
sx2 = np.linspace(0, 20, 2000)
spline2 = interpolate.UnivariateSpline(x2, y2, s=8)
sy2 = spline2(sx2)
plt.figure(figsize=(8, 5))
plt.subplot(211)
plt.plot(x1, y1, ".", label=u"数据点")
plt.plot(sx1, sy1, label=u"spline曲线")
plt.legend()
plt.subplot(212)
plt.plot(x2, y2, ".", label=u"数据点")
plt.plot(sx2, sy2, linewidth=2, label=u"spline曲线")
plt.plot(x2, np.sin(x2), label=u"无噪声曲线")
plt.legend()
#计算曲线和横线的交点
def root_at(self, v):
    coeff = self.get_coeffs()
    coeff -= v
    try:
        root = self.roots()
        return root
    finally:
        coeff += v
interpolate.UnivariateSpline.roots_at = root_at
plt.plot(sx2, sy2, lw = 2, label=u"spline曲线")
ax = plt.gca()
for level in [0.5, 0.75, -0.5, -0.75]:
    ax.axhline(level, ls=":", color="k")
    xr = spline2.roots_at(level)
    plt.plot(xr, spline2(xr), "ro")
plt.show()
#plt.savefig("c:\\figure1.png")