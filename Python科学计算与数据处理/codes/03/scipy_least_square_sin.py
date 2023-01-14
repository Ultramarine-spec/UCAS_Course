# -*- coding: utf-8 -*-
"""
使用leastsq()对带噪声的正弦波数据进行拟合。拟合所得到的参数虽然和实际的参数完全
不同，但是由于正弦函数具有周期性，如图所示，实际上拟合的结果和实际的函数是一致的。
"""
import numpy as np
from scipy import optimize

def func(x, A, k, theta):
    return A*np.sin(2*np.pi*k*x+theta)

#x = np.linspace(-2*np.pi, 0, 100)   
x = np.linspace(0, 2*np.pi, 100)
A, k, theta = 10, 0.34, np.pi/6 # 真实数据的函数参数
y0 = func(x, A, k, theta) # 真实数据
# 加入噪声之后的实验数据
y1 = y0 + 2 * np.random.randn(len(x)) 

#p0 = [10, 0.4, 0] # 第一次猜测的函数拟合参数
#p0 = [10, 1, 0] 
p0 = [7, 0.4, 0]

# 调用leastsq进行数据拟合
# residuals为计算误差的函数
# p0为拟合参数的初始值
# args为需要拟合的实验数据

popt, pcov = optimize.curve_fit(func, x, y1, p0=p0)

print u"真实参数:", [A, k, theta] 
print popt



import pylab as pl
pl.plot(x, y0, label=u"真实数据")  
#pl.plot(x, y1, label=u"带噪声的实验数据")  
pl.plot(x, y1, "o", label=u"带噪声的实验数据")
pl.plot(x, func(x, popt[0],popt[1],popt[2]), label=u"拟合数据")
pl.legend()
pl.show()
