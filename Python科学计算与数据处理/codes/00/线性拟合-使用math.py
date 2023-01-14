# -*- coding: utf-8 -*-
import math
def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r

if __name__ == '__main__':
    X=[ 1 ,2  ,3 ,4 ,5 ,6]
    Y=[ 2.5 ,3.51 ,4.45 ,5.52 ,6.47 ,7.51]
    a,b,r=linefit(X,Y)
    print("X=",X)
    print("Y=",Y)
    print("拟合结果: y = %10.5f x + %10.5f , r=%10.5f" % (a,b,r) )