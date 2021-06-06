import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print('loading data ...')
out = np.load('./features/mnist_9_out.npy')
x = np.load('./features/mnist_9_x.npy')
label = np.load('./features/mnist_9_label.npy')

print('tsne ...')
tsne = TSNE(n_components=2)
out_2d = tsne.fit_transform(out)
np.save('./features/mnist_9_out_2d.npy', out_2d)

print('plot ...')
plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(out_2d[:, 0], out_2d[:, 1], c=label)
plt.show()