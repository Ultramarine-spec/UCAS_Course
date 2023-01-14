# -*- coding: utf-8 -*-
"""
用各种计算函数最小值的fmin*()求卷积的逆运算.
"""
import scipy.optimize as opt 
import numpy as np 

def test_fmin_convolve(fminfunc, x, h, y, yn, h0): 
    """
    x (*) h = y, (*)表示卷积
    yn为在y的基础上添加一些干扰噪声的结果
    h0为求解h的初始值
    """
    def convolve_func(h): 
        """
        计算 yn - x (*) h 的power
        fmin将通过计算使得此power最小
        """ 
        return np.sum((yn - np.convolve(x, h))**2) 

    # 调用fmin函数，以x0为初始值
    hn = fminfunc(convolve_func, h0) 

    print fminfunc.__name__ 
    print "---------------------" 
    # 输出 x (*) hn 和 y 之间的相对误差
    print "error of y:", np.sum((np.convolve(x, hn)-y)**2)/np.sum(y**2) 
    # 输出 hn 和 h 之间的相对误差
    print "error of h:", np.sum((hn-h)**2)/np.sum(h**2) 
    print 

def test_n(m, n, nscale): 
    """
    随机产生x, h, y, yn, h0等数列，调用各种fmin函数求解b
    m为x的长度, n为h的长度, nscale为干扰的强度
    """
    x = np.random.rand(m) 
    h = np.random.rand(n) 
    y = np.convolve(x, h) 
    yn = y + np.random.rand(len(y)) * nscale
    h0 = np.random.rand(n) 

    test_fmin_convolve(opt.fmin, x, h, y, yn, h0) 
    test_fmin_convolve(opt.fmin_powell, x, h, y, yn, h0) 
    test_fmin_convolve(opt.fmin_cg, x, h, y, yn, h0)
    test_fmin_convolve(opt.fmin_bfgs, x, h, y, yn, h0)

if __name__ == "__main__":
    test_n(200, 20, 0.1) 
