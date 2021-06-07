import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print('loading data ...')
out = np.load('../features/mnist_out.npy')
x = np.load('../features/mnist_x.npy')
label = np.load('../features/mnist_label.npy')

print('tsne ...')
tsne = TSNE(n_components=2)
x_2d = tsne.fit_transform(x)
np.save('../features/mnist_x_2d.npy', x_2d)

print('plot ...')
plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(x_2d[:, 0], x_2d[:, 1], c=label)
plt.show()