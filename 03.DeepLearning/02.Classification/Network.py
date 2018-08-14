# coding=utf-8
import torch

from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.models import alexnet, vgg16, vgg19, resnet50, densenet121
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.functional import softmax

from lib.models.cifar import SENet
from lib.dataset.pytorch_dataset import Cifar10DataSetForPytorch
from lib.ProgressBar import ProgressBar

EPOCH = 20
BATCH_SIZE = 64
LR = 0.001
MODEL = "senet"
GPU_NUMS = 0

MODEL_LIST = [
    {
        "name": "alexnet",
        "model": alexnet,
        "pretrained" : False,
        "transform" : Compose([Resize(256),
                               RandomCrop(224),
                               RandomHorizontalFlip(),
                               ToTensor(),
                               Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])
                                                      ])
    },
    {
        "name": "vgg16",
        "model": vgg16,
        "pretrained" : False,
        "transform" : ToTensor()
    },
    {
        "name": "vgg19",
        "model": vgg19,
        "pretrained" : False,
        "transform" : ToTensor()
    },
    {
        "name" : "resnet50",
        "model" : resnet50,
        "pretrained" : False,
        "transform" : ToTensor()
    },
    {
        "name" : "densenet121",
        "model" : densenet121,
        "pretrained" : False,
        "transform" : ToTensor()
    },
    {
        "name" : "senet",
        "model" : SENet,
        "pretrained" : False,
        "transform" : ToTensor()
    }
]

# 准备数据
json = MODEL_LIST["name" == "senet"]

train_data = Cifar10DataSetForPytorch(train=True, transform=json["transform"])
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 准备网络
model = json["model"](json["pretrained"])

model = torch.nn.DataParallel(model).cuda() if GPU_NUMS > 1 else torch.nn.DataParallel(model)
optimizer = Adam(model.parameters(), lr=LR)
loss_func = CrossEntropyLoss().cuda() if GPU_NUMS > 0 else CrossEntropyLoss()

# 训练数据
proBar = ProgressBar(EPOCH, len(train_loader), "loss:%.3f,acc:%.3f")
for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        data = Variable(x.cuda() if GPU_NUMS > 0 else x)
        label = Variable(torch.squeeze(y, dim=1).type(torch.LongTensor).cuda() if GPU_NUMS > 0 else torch.squeeze(y, dim=1).type(torch.LongTensor))
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        prediction = torch.max(softmax(output), 1)[1]
        pred_label = prediction.data.cpu().numpy().squeeze()
        target_y = label.data.cpu().numpy()
        accuracy = sum(pred_label == target_y) / len(target_y)

        proBar.show(loss.data[0], accuracy)
torch.save(model.state_dict(), "%s.pth" % MODEL)