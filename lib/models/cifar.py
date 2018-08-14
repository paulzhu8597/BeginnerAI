# coding=utf-8
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential,AdaptiveAvgPool2d, Linear, Sigmoid

from torch.nn.init import kaiming_normal, constant

from lib.dataset.BasicDataSet import Cifar10DataSet
class CifarSEBasicBlock(Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = Sequential(
                Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out

    def _conv3x3(self,in_planes, out_planes, stride=1):
        return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SENet(Module):
    def __init__(self, n_size = 3,  block = CifarSEBasicBlock, config=Cifar10DataSet(), num_classes=10, reduction=16):
        super(SENet, self).__init__()
        self.inplane = 16
        self.config = config
        self.conv1 = Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(self.inplane)
        self.relu = ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal(m.weight)
            elif isinstance(m, BatchNorm2d):
                constant(m.weight, 1)
                constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SELayer(Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channel, channel // reduction),
            ReLU(inplace=True),
            Linear(channel // reduction, channel),
            Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y