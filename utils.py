import os
import torch
from torch import nn
import pandas as pd
import numpy as np


class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()

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

        # 32 * 8 * 8
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        print(x.shape)
        return x, x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 64)
        x = self.fc1(x)
        out = self.fc2(x)
        return x, out