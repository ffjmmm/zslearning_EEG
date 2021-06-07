import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import utils


BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ZERO_SHOT = True

# data_test = utils.MNIST_9(root='./data', train=False, transform=transforms.ToTensor())
data_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

cnn = utils.CNN_MNIST(ZERO_SHOT)
model_name = 'mnist_9_cnn.pkl'
cnn.load_state_dict(torch.load('../model/' + model_name))
cnn.to(device)

cnn.eval()

feature_x = []
feature_label = []

for step, (data, label) in enumerate(dataloader_test):
    label = label.data.numpy()[:, np.newaxis]
    data = data.to(device)
    x, _ = cnn(data)
    x = x.cpu().data.numpy()

    feature_x.append(x)
    feature_label.append(label)

    # pred = torch.max(out, 1)[1].data.numpy()
    # correct += int((pred == label).astype(int).sum())

array_x = np.concatenate(feature_x, axis=0)
array_label = np.concatenate(feature_label, axis=0)
# np.save('./features/mnist_zsl_x.npy', array_x)
# np.save('./features/mnist_zsl_label.npy', array_label)

print('tsne ...')
tsne = TSNE(n_components=2)
x_2d = tsne.fit_transform(array_x)
# np.save('./features/mnist_zsl_x_2d.npy', x_2d)

print('plot ...')
plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(x_2d[:, 0], x_2d[:, 1], c=array_label)
plt.show()