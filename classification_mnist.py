import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import utils

BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 50
LR = 0.001

transform = transforms.ToTensor()

data_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform)
data_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

cnn = utils.CNN_MNIST().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    train_loss = 0.0
    for step, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        _, out = cnn(data)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy()

    acc = 0.
    for step, (data, label) in enumerate(dataloader_test):
        data = data.to(device)
        label = F.one_hot(label, num_classes=10).float()
        _, out = cnn(data)
        out = out.cpu()
        pred = torch.max(out, 1)[1].data.numpy()
        label = torch.max(label, 1)[1].data.numpy()
        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        acc += accuracy
    acc /= (step + 1)

    print('%d: %.4f %.4f' % (epoch, train_loss, acc))

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.subplots()
    # ax.set_title('Epoch %d' % epoch)
    # tsne = TSNE(n_components=2)
    #
    # for data, label in dataloader_test:
    #     data = data.to(device)
    #     x, _ = cnn(data)
    #     x = x.cpu().data.numpy()
    #     x = tsne.fit_transform(x)
    #     ax.scatter(x[:, 0], x[:, 1], c=label, cmap='rainbow')
    #     break
    #
    # plt.show()