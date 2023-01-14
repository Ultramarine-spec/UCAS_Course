# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
x = np.random.rand(10)
y = np.random.rand(10)
np.random.shuffle(y)
plt.plot(x, y, 'o')
for s in (0, 1e-1):
    tck, t = interpolate.splprep([x, y], s=s)
    #首先调用splprep()，其第一个参数为一组一维数组，每个数组是各点在
    #对应轴上的坐标s为平滑系数。splprep()返回两个对象，其中tck是一个
    #元祖，它包含了插值曲线的所有信息，t是自动计算出参数曲线的参数数组。
    
    xi,yi=interpolate.splev(np.linspace(t[0],t[-1],200),tck)
     #调用splev()进行插值运算，其第一个参数为一个新的参数数组，第二个为
     #splprep()返回的第一个对象。
    # plt.plot(xi, yi, lw=2, label=u"s=%g" %s)
  
    if s == 0:
        plt.plot(xi, yi, lw=1, label=u"s=%g" %s)
    else:
        plt.plot(xi, yi, lw=3, alpha=0.4, label=u"s=%g" %s)
plt.legend()
plt.show()