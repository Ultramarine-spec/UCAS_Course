import time
import numpy as np
import matplotlib.pyplot as plt


def triangle_wave(x, c=0.6, c0=0.4, hc=1.0):
    x = x - int(x)
    if x >= c:
        r = 0.0
    elif x < c0:
        r = x / c0 * hc
    else:
        r = (c - x) / (c - c0) * hc
    return r


x = np.linspace(0, 2, 1000000)
start = time.process_time()
y1 = np.array([triangle_wave(t) for t in x])
print("y1 time:", time.process_time() - start)

triangle_ufunc1 = np.frompyfunc(triangle_wave, 4, 1)
start = time.process_time()
y2 = triangle_ufunc1(x, 0.6, 0.4, 1.0)
y2 = y2.astype(np.float64)
print("y2 time:", time.process_time() - start)

triangle_ufunc2 = np.frompyfunc(lambda i: triangle_wave(i, 0.6, 0.4, 1.0), 1, 1)
start = time.process_time()
y3 = triangle_ufunc2(x)
y3 = y3.astype(np.float64)
print("y3 time:", time.process_time() - start)

triangle_ufunc3 = np.vectorize(triangle_wave, otypes=[np.float64])
start = time.process_time()
y4 = triangle_ufunc3(x)
print("y4 time:", time.process_time() - start)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x, y1, label="y1")
plt.subplot(2, 2, 2)
plt.plot(x, y2, label="y2")
plt.subplot(2, 2, 3)
plt.plot(x, y3, label="y3")
plt.subplot(2, 2, 4)
plt.plot(x, y4, label="y4")
plt.show()
