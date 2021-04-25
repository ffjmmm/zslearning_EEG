from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from PyNomaly import loop
import random

n_centers = 7
x, y = make_blobs(n_samples=500, centers=n_centers, n_features=2, random_state=0)
x = np.array(x)

random.seed(0)
x_random = [[random.uniform(-10., 10.), random.uniform(-10., 10.)]
            for _ in range(30)]
x_random = np.array(x_random)

fig = plt.figure(figsize=(7, 7))
ax_1 = fig.add_subplot(211)
ax_1.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')
ax_1.scatter(x_random[:, 0], x_random[:, 1], c='black')

x = np.vstack((x, x_random))
scores = loop.LocalOutlierProbability(x).fit().local_outlier_probabilities

ax_2 = fig.add_subplot(212)
ax_2.scatter(x[:, 0], x[:, 1], c=scores, cmap='seismic')

plt.show()