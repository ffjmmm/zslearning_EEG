import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import utils


BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 10
LR = 0.001
ZERO_SHOT = True

if ZERO_SHOT:
    data_train = utils.MNIST_9(root='./data', train=True, transform=transforms.ToTensor())
    data_test = utils.MNIST_9(root='./data', train=False, transform=transforms.ToTensor())
else:
    data_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    data_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
dataloader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

cnn = utils.CNN_MNIST(ZERO_SHOT).to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    cnn.train()
    for step, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        _, out = cnn(data)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if step % 100 == 0:
        #     print('Train Epoch %d Step %d loss: %.4f' % (epoch, step, loss.cpu().data.numpy()))

    cnn.eval()
    test_loss = 0.0
    correct = 0
    for step, (data, label) in enumerate(dataloader_test):
        data, label = data.to(device), label.to(device)
        _, out = cnn(data)
        loss = loss_func(out, label)
        test_loss += loss.cpu().data.numpy()
        out = out.cpu()
        label = label.cpu().data.numpy()
        pred = torch.max(out, 1)[1].data.numpy()
        correct += int((pred == label).astype(int).sum())

    print('Epoch %d loss: %.4f accuracy: %.4f' % (epoch, test_loss, correct / len(dataloader_test.dataset)))

print('model save ...')
model_name = 'mnist_9_cnn.pkl' if ZERO_SHOT else 'mnist_cnn.pkl'
torch.save(cnn.state_dict(), './model/' + model_name)

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