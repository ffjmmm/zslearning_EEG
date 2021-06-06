import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

import utils


BATCH_SIZE = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ZERO_SHOT = True

if ZERO_SHOT:
    data_train = utils.MNIST_9(root='./data', train=True, transform=transforms.ToTensor())
else:
    data_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

cnn = utils.CNN_MNIST(ZERO_SHOT)
model_name = 'mnist_9_cnn.pkl' if ZERO_SHOT else 'mnist_cnn.pkl'
cnn.load_state_dict(torch.load('./model/' + model_name))
cnn.to(device)

feature_x = []
feature_out = []
feature_label = []

cnn.eval()
for data, label in dataloader_train:
    data, label = data.to(device), label.to(device)
    x, out = cnn(data)

    feature_out.append(out.cpu().data.numpy())
    feature_x.append(x.cpu().data.numpy())
    feature_label.append(label.cpu().data.numpy()[:, np.newaxis])

array_out = np.concatenate(feature_out, axis=0)
array_x = np.concatenate(feature_x, axis=0)
array_label = np.concatenate(feature_label, axis=0)

print(array_out.shape)
print(array_x.shape)
print(array_label.shape)

if ZERO_SHOT:
    np.save('./features/mnist_9_out.npy', array_out)
    np.save('./features/mnist_9_x.npy', array_x)
    np.save('./features/mnist_9_label.npy', array_label)
else:
    np.save('./features/mnist_out.npy', array_out)
    np.save('./features/mnist_x.npy', array_x)
    np.save('./features/mnist_label.npy', array_label)
