import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets

import matplotlib.pyplot as plt


def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], str(label[i]),
                color=plt.cm.Set1(label[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    return fig


digits = datasets.load_digits(n_class=10)
data = digits.data
label = digits.target
n_samples, n_features = data.shape
print(n_samples, n_features)

tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(data)

fig = plot_embedding(result, label)
plt.show()
