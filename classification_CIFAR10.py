import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

BATCH_SIZE = 32

data_train = torchvision.datasets.CIFAR10(root='./data', train=True)
data_test = torchvision.datasets.CIFAR10(root='./data', train=False)
dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        