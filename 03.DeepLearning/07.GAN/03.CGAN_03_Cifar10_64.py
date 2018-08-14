import torch
import torchvision

from torch.nn import BCELoss, Module, Conv2d, ConvTranspose2d, Linear, BatchNorm2d, ReLU, LeakyReLU,Sigmoid, Tanh,DataParallel
from torchvision.transforms import Resize, Compose,ToTensor,Normalize
from torch.optim import Adam
from torch.autograd import Variable

from lib.dataset.pytorch_dataset import Cifar10DataSetForPytorch
from lib.ProgressBar import ProgressBar

CONFIG = {
    "DATA_PATH" : "/input/cifar10/", #"/input/Faces/Eyeglasses",
    "EPOCHS" : 100,
    "BATCH_SIZE" : 64,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 64,
    "IMAGE_CHANNEL" : 3,
    "NOISE_DIM" : 100,
    "LEARNING_RATE" : 2e-4,
    "BETA1" : 0.5
}

class Generator(Module):
    def __init__(self, feature_num=64):
        super(Generator, self).__init__()
        self.feature_num = feature_num
        self.gc_full = Linear(CONFIG["NOISE_DIM"] + 10, CONFIG["NOISE_DIM"] + 10)
        self.deconv1 = ConvTranspose2d(CONFIG["NOISE_DIM"] + 10, feature_num * 8,
                                       kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(feature_num * 8)

        self.deconv2 = ConvTranspose2d(feature_num * 8, feature_num * 4,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(feature_num * 4)

        self.deconv3 = ConvTranspose2d(feature_num * 4, feature_num * 2,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = BatchNorm2d(feature_num * 2)

        self.deconv4 = ConvTranspose2d(feature_num * 2, feature_num,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = BatchNorm2d(feature_num)

        self.deconv5 = ConvTranspose2d(feature_num, CONFIG["IMAGE_CHANNEL"], kernel_size=4,
                                       stride=2, padding=1, bias=False)
    def forward(self, input, label):
        network = torch.cat([input, label], dim=1)
        network = self.gc_full(network)
        network = network.view(-1, CONFIG["NOISE_DIM"] + 10, 1, 1)
        network = self.deconv1(network)
        network = self.bn1(network)
        network = ReLU()(network)

        network = self.deconv2(network)
        network = self.bn2(network)
        network = ReLU()(network)

        network = self.deconv3(network)
        network = self.bn3(network)
        network = ReLU()(network)

        network = self.deconv4(network)
        network = self.bn4(network)
        network = ReLU()(network)

        network = self.deconv5(network)
        network = Tanh()(network)
        return network

class Discriminator(Module):
    def __init__(self, features_num=64):
        super(Discriminator, self).__init__()
        self.features_num = features_num
        self.input_conv1 = Conv2d(in_channels=CONFIG["IMAGE_CHANNEL"], out_channels=features_num,
                                  kernel_size=4, stride=2, padding=1, bias=False)
        self.input_conv2 = Conv2d(in_channels=features_num, out_channels=features_num * 2,
                                  kernel_size=4, stride=2, padding=1, bias=False)
        self.input_conv3 = Conv2d(features_num * 2, features_num * 4,
                                  kernel_size=4, stride=2, padding=1, bias=False)
        self.input_bn3 =  BatchNorm2d(features_num * 4)
        self.input_conv4 = Conv2d(features_num * 4, features_num * 8,
                                  kernel_size=4, stride=2, padding=1, bias=False)
        self.input_bn4 =  BatchNorm2d(features_num * 8)
        self.input_conv5 = Conv2d(features_num * 8, features_num * 8, kernel_size=4, stride=1, padding=0, bias=False)

        self.cat_conv1 = Linear(512 +10, 256)
        self.cat_merge = Linear(256 , 1)
    def forward(self, input, label):
        network_1 = self.input_conv1(input)
        network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
        network_1 = self.input_conv2(network_1)
        network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
        network_1 = self.input_conv3(network_1)
        network_1 = self.input_bn3(network_1)
        network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
        network_1 = self.input_conv4(network_1)
        network_1 = self.input_bn4(network_1)
        network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
        network_1 = self.input_conv5(network_1)
        network_1 = network_1.view(-1, self.features_num * 8)
        network= torch.cat( [ network_1 , label ], 1)
        network = self.cat_conv1(network)
        network = LeakyReLU(negative_slope=0.2, inplace=True)(network)
        network = self.cat_merge(network)
        network = Sigmoid()(network)

        return network

NetG = Generator()
NetD = Discriminator()
BCE_LOSS = BCELoss()
G_optimizer = Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))
D_optimizer = Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    BCE_LOSS = BCE_LOSS.cuda()

if CONFIG["GPU_NUMS"] > 1:
    NetG = DataParallel(NetG, device_ids=[0,1])
    NetD = DataParallel(NetD, device_ids=[0, 1])

transform = Compose([
    Resize(CONFIG["IMAGE_SIZE"]),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3)
])
train_loader = torch.utils.data.DataLoader(
    Cifar10DataSetForPytorch(root=CONFIG["DATA_PATH"], train=True, transform=transform),
    # ImageFolder(root=CONFIG["DATA_PATH"], transform=transform),
    batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y

Predict_Noise_var = Variable(torch.randn(100, CONFIG["NOISE_DIM"]).cuda() if CONFIG["GPU_NUMS"] > 0 else torch.randn(100, CONFIG["NOISE_DIM"]))

temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
Predict_y = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    Predict_y = torch.cat([Predict_y, temp], 0)

Predict_y = one_hot(Predict_y.long())
Predict_y = Variable(Predict_y.cuda() if CONFIG["GPU_NUMS"] > 0 else Predict_y)

bar = ProgressBar(CONFIG["EPOCHS"], len(train_loader), "D Loss:%.3f, G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    if epoch % 30 == 0:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10

    for img_real, label_real in train_loader:
        mini_batch = label_real.shape[0]

        label_true = torch.ones(mini_batch)
        label_false = torch.zeros(mini_batch)
        label_true_var = Variable(label_true.cuda() if CONFIG["GPU_NUMS"] > 0 else label_true)
        label_false_var = Variable(label_false.cuda() if CONFIG["GPU_NUMS"] > 0 else label_false)

        label = one_hot(label_real.long().squeeze())
        image_var = Variable(img_real.cuda() if CONFIG["GPU_NUMS"] > 0 else img_real)
        label_var = Variable(label.cuda() if CONFIG["GPU_NUMS"] > 0 else label)

        NetD.zero_grad()
        D_real = NetD(image_var, label_var)
        D_real_loss = BCE_LOSS(D_real, label_true_var)

        Noise_var = Variable(torch.randn(mini_batch, CONFIG["NOISE_DIM"]).cuda() if CONFIG["GPU_NUMS"] > 0 else torch.randn(mini_batch, CONFIG["NOISE_DIM"]))
        image_fake = NetG(Noise_var, label_var)
        D_fake = NetD(image_fake, label_var)
        D_fake_loss = BCE_LOSS(D_fake, label_false_var)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()

        NetG.zero_grad()
        Noise_var = Variable(torch.randn(mini_batch, CONFIG["NOISE_DIM"]).cuda() if CONFIG["GPU_NUMS"] > 0 else torch.randn(mini_batch, CONFIG["NOISE_DIM"]))
        image_fake = NetG(Noise_var,label_var)
        D_fake = NetD(image_fake,label_var)

        G_loss = BCE_LOSS(D_fake, label_true_var)

        G_loss.backward()
        G_optimizer.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    test_images = NetG(Predict_Noise_var, Predict_y)

    torchvision.utils.save_image(test_images.data[:100],'outputs/Cifar10_%03d.png' % (epoch),nrow=10,
                                 normalize=True,range=(-1,1), padding=0)