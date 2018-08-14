# coding=utf-8
import torch
from torch import nn as nn
from torch.utils import data as data
from torch.autograd import Variable
import torchvision
from torch.optim import Adam
from torch.nn import functional as F

from lib.dataset.pytorch_dataset import MNISTDataSetForPytorch
from lib.ProgressBar import ProgressBar

torch.manual_seed(1)
EPOCH = 10
LR = 0.001
GPU_NUMS = 10


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5,stride=1,padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5,stride=1,padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

train_data = MNISTDataSetForPytorch(train=True, transform=torchvision.transforms.ToTensor())
train_loader = data.DataLoader(dataset=train_data, batch_size=128,
                               shuffle=True)

cnn = CNN().cuda() if GPU_NUMS > 0 else CNN()
optimizer = Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss().cuda() if GPU_NUMS > 0 else nn.CrossEntropyLoss()
proBar = ProgressBar(EPOCH, len(train_loader), "Loss: %.3f;Accuracy: %.3f")
for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x.cuda() if GPU_NUMS > 0 else x)
        b_y = Variable(y.type(torch.LongTensor).cuda() if GPU_NUMS > 0 else y.type(torch.LongTensor)).squeeze_()
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = torch.max(F.softmax(output, dim=1), 1)[1]
        pred_y = prediction.cpu().data.numpy().squeeze()
        target_y = b_y.cpu().data.numpy()
        accuracy = sum(pred_y == target_y) / len(target_y)

        proBar.show(loss.data[0], accuracy)

test_x = Variable(torch.unsqueeze(torch.FloatTensor(train_data.test_data), dim=1).cuda() if GPU_NUMS > 0 else torch.unsqueeze(torch.FloatTensor(train_data.test_data), dim=1))
test_y = Variable(torch.LongTensor(train_data.test_labels).cuda() if GPU_NUMS > 0 else torch.LongTensor(train_data.test_labels))
test_y = test_y.squeeze()
test_output = cnn(test_x)
pred_y = torch.max(F.softmax(test_output, dim=1), 1)[1].cpu().data.numpy().squeeze()
target_y = test_y.cpu().data.numpy()
accuracy = sum(pred_y == target_y) / len(target_y)
print("test accuracy is %.3f" % accuracy)