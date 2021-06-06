import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PyNomaly import loop
from sklearn.manifold import TSNE

import utils

BATCH_SIZE = 1
NUM_X = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print('load data ...')
x = np.load('./features/mnist_zsl_x.npy')

data_test = utils.MNIST_1(root='./data', train=False, transform=transforms.ToTensor())
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

print('load model ...')
cnn = utils.CNN_MNIST(True)
model_name = 'mnist_9_cnn.pkl'
cnn.load_state_dict(torch.load('./model/' + model_name))
cnn.to(device)

tsne = TSNE(n_components=2)

cnn.eval()

x_all = []
scores_all = []

k = 0

for data, _ in dataloader_test:
    k += BATCH_SIZE
    if k > NUM_X:
        break
    data = data.to(device)
    x_ = cnn(data)[0].cpu().data.numpy()
    x_loop = np.vstack((x, x_))
    Loop = loop.LocalOutlierProbability(x_loop, progress_bar=True, n_neighbors=30, use_numba=True)
    scores = Loop.fit().local_outlier_probabilities

    x_all.append(x_)
    scores_all.append(scores[-BATCH_SIZE:, np.newaxis])

x_all = np.concatenate(x_all, axis=0)
scores_all = np.concatenate(scores_all, axis=0)
np.save('./features/mnist_loop_x.npy', x_all)
np.save('./features/mnist_loop_scores.npy', scores_all)

n_ = len(x_all)

print('tsne ...')
x = np.vstack((x_all, x))
x_2d = tsne.fit_transform(x)

plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(x_2d[n_:, 0], x_2d[n_:, 1], c='black')
plt.scatter(x_2d[:n_, 0], x_2d[:n_, 1], c=scores_all, cmap='seismic')
plt.show()