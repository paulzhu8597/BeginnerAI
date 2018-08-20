import torch
import torchvision

from torch.nn import BCELoss, Module, Conv2d, ConvTranspose2d, Linear, BatchNorm2d, ReLU, LeakyReLU,Sigmoid
from torchvision.transforms import Resize, Compose,ToTensor
from torch.optim import Adam
from torch.autograd import Variable

from lib.dataset.pytorch_dataset import Cifar10DataSetForPytorch
from lib.ProgressBar import ProgressBar

CONFIG = {
    "DATA_PATH" : "/input/cifar10/",
    "EPOCHS" : 100,
    "BATCH_SIZE" : 128,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 32,
    "IMAGE_CHANNEL" : 3,
    "NOISE_DIM" : 100,
    "LEARNING_RATE" : 1e-3,
    "BETA1" : 0.5
}

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gc_full = Linear(CONFIG["NOISE_DIM"] + 10, 512)

        self.dconv1 = ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.dconv2 = ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.dconv3 = ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.dconv4 = ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=1)

        self.bn4 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn2 = BatchNorm2d(256)
        self.bn1 = BatchNorm2d(512)

    def forward(self, input, label):
        network = torch.cat([input, label], dim=1)

        network = self.gc_full(network)
        network = ReLU(inplace=True)(network)
        network = network.view(-1, 512, 1, 1)
        network = ReLU()(self.bn2(self.dconv1(network)))
        network = ReLU()(self.bn3(self.dconv2(network)))
        network = ReLU()(self.bn4(self.dconv3(network)))
        network = Sigmoid()(self.dconv4(network))

        return network

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2)

        self.bn1 = BatchNorm2d(64)
        self.bn2 = BatchNorm2d(128)
        self.bn3 = BatchNorm2d(256)
        self.bn4 = BatchNorm2d(512)

        self.d_fc = Linear(512 +10, 256)

        self.merge_layer = Linear(256 , 1)

    def forward(self, input, label):
        network = self.conv1(input)
        network = LeakyReLU(negative_slope=0.2)(network)
        network = self.bn1(network)

        network = self.conv2(network)
        network = LeakyReLU(negative_slope=0.2)(network)
        network = self.bn2(network)

        network = self.conv3(network)
        network = LeakyReLU(negative_slope=0.2)(network)
        network = self.bn3(network)

        network = self.conv4(network)
        network = LeakyReLU(negative_slope=0.2)(network)
        network = self.bn4(network)

        network = network.view(-1, 512)
        network= torch.cat( [ network , label ], 1)

        network = self.d_fc(network)
        network = LeakyReLU(negative_slope=0.2)(network)
        network = self.merge_layer(network)
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

transform = Compose([
    ToTensor()
])
train_loader = torch.utils.data.DataLoader(
    Cifar10DataSetForPytorch(root=CONFIG["DATA_PATH"], train=True, transform=transform),
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
    if epoch % 10 == 0:
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