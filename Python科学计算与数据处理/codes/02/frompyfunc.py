import time
import numpy as np
from triangle_wave import *

x = np.linspace(0, 2, 1000)

start = time.clock()
y1 = np.array([triangle_wave(t, 0.6, 0.4, 1.0) for t in x])
print "y1 loop:", time.clock() - start

start = time.clock()
triangle_ufunc1 = np.frompyfunc(triangle_wave, 4, 1)
y2 = triangle_ufunc1(x, 0.6, 0.4, 1.0)
print "y2 loop:", time.clock() - start

start = time.clock()
triangle_ufunc2 = np.frompyfunc( lambda x: triangle_wave(x, 0.6, 0.4, 1.0), 1, 1)
y3 = triangle_ufunc2(x)
print "y3 loop:", time.clock() - start

start = time.clock()
triangle_ufunc3 = np.vectorize(triangle_wave, otypes=[np.float])
y4 = triangle_ufunc3(x, 0.6, 0.4, 1.0) 
print "y4 loop:", time.clock() - start