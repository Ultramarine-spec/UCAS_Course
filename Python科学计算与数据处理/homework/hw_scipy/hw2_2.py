from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def f(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


p0 = [1, 1, 1]
x0 = np.linspace(0, 10, 100)
y0 = f(x0, 1, 5, 2)
np.random.seed(42)
y1 = y0 + 0.1 * np.random.randn(len(x0))
p_origin, _ = curve_fit(f, x0, y0, p0=p0)
p_noisy, _ = curve_fit(f, x0, y1, p0=p0)
print("拟合结果：", p_origin)
print("加入高斯噪声后的拟合结果：", p_noisy)

plt.figure()
plt.plot(x0, y0, label="f(x)")
plt.plot(x0, y1, label="f(x) + noisy")
plt.legend()
plt.show()
plt.figure()
plt.plot(x0, f(x0, *p_origin), label="fit_origin")
plt.plot(x0, f(x0, *p_noisy), label="fit_noisy")
plt.legend()
plt.show()
