import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
shape, scale = 1, 1
s = np.random.gamma(shape, scale, 1000)
plt.figure()
count, bins, ignored = plt.hist(s, bins=50, density=True)

x = np.linspace(0, 10, 1000)
y = stats.gamma.pdf(x, 1, scale=1)
plt.plot(x, y)
plt.show()
