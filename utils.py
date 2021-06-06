import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision


class CNN_MNIST(nn.Module):
    def __init__(self, zero_shot):
        super(CNN_MNIST, self).__init__()

        # input 1 x 28 x 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        # 16 * 14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        # 32 * 7 * 7
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 49, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 9 if zero_shot else 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 49)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return x, out


class MNIST_9(torch.utils.data.Dataset):
    def __init__(self, root, train, transform):
        super(MNIST_9, self).__init__()

        if train:
            data = torchvision.datasets.MNIST(root=root, train=True, transform=transform)
        else:
            data = torchvision.datasets.MNIST(root=root, train=False, transform=transform)

        label = data.targets.numpy()
        data = data.data.numpy()
        index = np.nonzero(label == 9)[0]
        self.data = torch.from_numpy(np.delete(data, index, axis=0)).type(torch.FloatTensor)
        self.data = self.data / 255.
        self.label = torch.from_numpy(np.delete(label, index))
        self.num = len(self.data)

    def __getitem__(self, item):
        return self.data[item].reshape(1, 28, 28), self.label[item]

    def __len__(self):
        return self.num


class MNIST_1(torch.utils.data.Dataset):
    def __init__(self, root, train, transform):
        super(MNIST_1, self).__init__()

        if train:
            data = torchvision.datasets.MNIST(root=root, train=True, transform=transform)
        else:
            data = torchvision.datasets.MNIST(root=root, train=False, transform=transform)

        label = data.targets.numpy()
        data = data.data.numpy()
        index = np.nonzero(label != 9)[0]
        self.data = torch.from_numpy(np.delete(data, index, axis=0)).type(torch.FloatTensor)
        self.data = self.data / 255.
        self.label = torch.from_numpy(np.delete(label, index))
        self.num = len(self.data)

    def __getitem__(self, item):
        return self.data[item].reshape(1, 28, 28), self.label[item]

    def __len__(self):
        return self.num
