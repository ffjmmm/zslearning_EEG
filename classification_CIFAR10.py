import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 10
LR = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
data_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataiter = iter(dataloader_train)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid((images)))
# print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # input 3 x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        # 32 * 16 * 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

for epoch in range(EPOCH):
    train_loss = 0.0
    for step, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        label = F.one_hot(label, num_classes=10).float()
        out = cnn(data)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy()

    acc = 0.
    for step, (data, label) in enumerate(dataloader_test):
        data = data.to(device)
        label = F.one_hot(label, num_classes=10).float()
        out = cnn(data).cpu()
        pred = torch.max(out, 1)[1].data.numpy()
        label = torch.max(label, 1)[1].data.numpy()
        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        acc += accuracy
    acc /= (step + 1)

    print('%d: %.4f %.4f' % (epoch, train_loss, acc))