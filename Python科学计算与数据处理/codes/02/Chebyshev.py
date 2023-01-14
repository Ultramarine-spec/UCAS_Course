# -*- coding: utf-8 -*-
import numpy as np
from numpy.polynomial import  Chebyshev

def f(x):
    return 1.0/ (1+25*x**2)

n = 11
x1 = np.linspace(-1, 1, n) 
x2 = Chebyshev.basis(n).roots() 
xd = np.linspace(-1,1, 200)

c1 = Chebyshev.fit(x1, f(x1), n - 1, domain=[-1,1])
c2 = Chebyshev.fit(x2, f(x2), n - 1, domain=[-1,1])

print u"插值多项式的最大误差：",
print u"等距离取样点：", abs(c1(xd) - f(xd)).max()
print u"契比雪夫节点：", abs(c2(xd) - f(xd)).max()
