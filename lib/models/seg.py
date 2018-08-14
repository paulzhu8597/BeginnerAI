import torch

from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Module, Dropout2d, BatchNorm2d, ConvTranspose2d
from torch.nn.functional import upsample, log_softmax, nll_loss, pad

class FCNs(Module):
    def __init__(self, cfg):
        super(FCN8s, self).__init__()
        self.CFG = cfg
        self.Conv_Block1 = Sequential(
            Conv2d(in_channels=self.CFG["IMAGE_CHANNEL"], out_channels=64, kernel_size=3, padding=100),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block4 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block5 = Sequential(
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.classifier = Sequential(
            Conv2d(in_channels=512, out_channels=4096, kernel_size=7),
            ReLU(inplace=True),
            Dropout2d(),
            Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            ReLU(inplace=True),
            Dropout2d(),
            Conv2d(in_channels=4096, out_channels=self.CFG["CLASS_NUMS"], kernel_size=1)
        )

    def forward(self, x):
        self.conv1_result = self.Conv_Block1(x)
        self.conv2_result = self.Conv_Block2(self.conv1_result)
        self.conv3_result = self.Conv_Block3(self.conv2_result)
        self.conv4_result = self.Conv_Block4(self.conv3_result)
        self.conv5_result = self.Conv_Block5(self.conv4_result)

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.Conv_Block1,
                  self.Conv_Block2,
                  self.Conv_Block3,
                  self.Conv_Block4,
                  self.Conv_Block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, Conv2d) and isinstance(l2, Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

class FCN8s(FCNs):
    def __init__(self, cfg):
        super(FCN8s, self).__init__(cfg=cfg)

        self.score_pool4 = Conv2d(512, self.n_classes, 1)
        self.score_pool3 = Conv2d(256, self.n_classes, 1)

    def forward(self, x):
        super(FCN8s, self).forward(x)

        score = self.classifier(self.conv5_result)
        score_pool4 = self.score_pool4(self.conv4_result)
        score_pool3 = self.score_pool4(self.conv3_result)

        score = upsample(score,size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4

        score = upsample(score,size=score_pool3.size()[2:], mode='bilinear')
        score += score_pool3

        output = upsample(score, size=x.size()[2:], mode='bilinear')
        return output

class FCN16s(FCNs):
    def __init__(self, cfg):
        super(FCN16s, self).__init__(cfg)

        self.score_pool4 = Conv2d(512, self.n_classes, 1)

    def forward(self, x):
        super(FCN16s, self).forward(x)

        score = self.classifier(self.conv5_result)
        score_pool4 = self.score_pool4(self.conv4_result)

        score = upsample(score,size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4
        output = upsample(score, size=x.size()[2:], mode='bilinear')
        return output

class FCN32s(FCNs):
    def __init__(self, cfg):
        super(FCN32s, self).__init__()

    def forward(self, x):
        super(FCN32s, self).forward(x)

        score = self.classifier(self.conv5_result)

        output = upsample(score, size=x.size()[2:], mode='bilinear')
        return output

class UNet(Module):
    def __init__(self, CFG):
        super(UNet, self).__init__()
        self.CFG = CFG
        self.is_deconv = True
        self.in_channel = CFG["IMAGE_CHANNEL"]
        self.is_batchnorm = True
        self.feature_scale = CFG["FEATURE_SCALE"]

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = MaxPool2d(kernel_size=2)

        self.center = UNetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3])
        self.up_concat3 = UNetUp(filters[3], filters[2])
        self.up_concat2 = UNetUp(filters[2], filters[1])
        self.up_concat1 = UNetUp(filters[1], filters[0])

        # final conv (without any concat)
        self.final = Conv2d(filters[0], CFG["CLASS_NUM"], 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

class UNetConv2(Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UNetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = Sequential(Conv2d(in_size, out_size, 3, 1, 0),
                                       BatchNorm2d(out_size),
                                       ReLU(),)
            self.conv2 = Sequential(Conv2d(out_size, out_size, 3, 1, 0),
                                       BatchNorm2d(out_size),
                                       ReLU(),)
        else:
            self.conv1 = Sequential(Conv2d(in_size, out_size, 3, 1, 0),
                                       ReLU(),)
            self.conv2 = Sequential(Conv2d(out_size, out_size, 3, 1, 0),
                                       ReLU(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UNetUp(Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.conv = UNetConv2(in_size, out_size, False)
        self.up = ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    log_p = log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.type(torch.FloatTensor).data.sum()
    return loss