import torch
import torchvision

from torch.nn import Module, Sequential, Conv2d, ELU, AvgPool2d, Linear, Tanh,L1Loss, Upsample
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.autograd import Variable
from torch.optim import Adam
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from lib.dataset.pytorch_dataset import Cifar10DataSetForPytorch
from lib.ProgressBar import ProgressBar

CONFIG = {
    ""
}

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
NOISE_DIM = 64
GPU_NUMS = 1
BATCH_SIZE = 128
EPOCHS = 100

def conv_block(in_dim,out_dim):
    return Sequential(Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                      ELU(True),
                      Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                      ELU(True),
                      Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                      AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
    return    Sequential(Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         ELU(True),
                         Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         ELU(True),
                         Upsample(scale_factor=2))
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Sequential(
            Conv2d(IMAGE_CHANNEL, 64, kernel_size=3, stride=1, padding=1),
            ELU(True),
            conv_block(64, 64)
        )

        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 192)

        self.conv4 = Sequential(
            Conv2d(192, 192, kernel_size=3, stride=1,padding=1),
            ELU(True),
            Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            ELU(True)
        )
        self.embed1 = Linear(192*4*4, 64)
        self.embed2 = Linear(64, 64*4*4)

        self.deconv1 = deconv_block(64,64)
        self.deconv2 = deconv_block(64,64)
        self.deconv3 = deconv_block(64,64)
        self.deconv4 = Sequential(Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                  ELU(True),
                                  Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                  ELU(True),
                                  Conv2d(64, IMAGE_CHANNEL, kernel_size=3, stride=1, padding=1),
                                  Tanh())
    def forward(self, x):
        network = self.conv1(x)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)

        network = network.view(network.size(0), 64*3*4*4)

        network = self.embed1(network)
        network = self.embed2(network)
        network = network.view(network.size(0), 64, 4, 4)
        network = self.deconv1(network)
        network = self.deconv2(network)
        network = self.deconv3(network)
        network = self.deconv4(network)
        return network

class Generator(Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.embed1 = Linear(NOISE_DIM, 64*4*4)

        # 8 x 8
        self.deconv1 = deconv_block(64, 64)
        # 16 x 16
        self.deconv2 = deconv_block(64, 64)
        # 32 x 32
        self.deconv3 = deconv_block(64, 64)
        self.deconv4 = Sequential(Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                  ELU(True),
                                  Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                  ELU(True),
                                  Conv2d(64, IMAGE_CHANNEL, kernel_size=3, stride=1, padding=1))


    def forward(self,x):
        out = self.embed1(x)
        out = out.view(out.size(0), 64, 4, 4)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out

Net_D=Discriminator()
Net_G = Generator()

if GPU_NUMS > 0:
    Net_D.cuda()
    Net_G.cuda()

###########   LOSS & OPTIMIZER   ##########
optimizerD = Adam(Net_D.parameters(),lr=0.0001, betas=(0.5, 0.999))
optimizerG = Adam(Net_G.parameters(),lr=0.0001, betas=(0.5, 0.999))

dataset = Cifar10DataSetForPytorch(train=True, transform=Compose(
    [
        ToTensor(),
        # Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ]))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

k = 0
proBar = ProgressBar(EPOCHS, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    for index, (images,_) in enumerate(train_loader):
        mini_batch = images.shape[0]
        noise = Variable(torch.FloatTensor(mini_batch, NOISE_DIM, 1, 1).cuda() if GPU_NUMS > 0 else torch.FloatTensor(mini_batch, NOISE_DIM, 1, 1))
        real = Variable(torch.FloatTensor(mini_batch, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE).cuda() if GPU_NUMS > 0 else torch.FloatTensor(mini_batch, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        label = Variable(torch.FloatTensor(1).cuda() if GPU_NUMS > 0 else torch.FloatTensor(1))

        Net_D.zero_grad()
        real.data.resize_(images.size()).copy_(images)

        # generate fake data
        noise.data.resize_(images.size(0), NOISE_DIM)
        noise.data.uniform_(-1,1)
        fake = Net_G(noise)

        fake_recons = Net_D(fake.detach())
        real_recons = Net_D(real)

        err_real = torch.mean(torch.abs(real_recons-real))
        err_fake = torch.mean(torch.abs(fake_recons-fake))

        errD = err_real - k*err_fake
        errD.backward()
        optimizerD.step()

        Net_G.zero_grad()
        fake = Net_G(noise)
        fake_recons = Net_D(fake)
        errG = torch.mean(torch.abs(fake_recons-fake))
        errG.backward()
        optimizerG.step()

        balance = (0.5 * err_real - err_fake).data[0]
        k = min(max(k + 0.001 * balance,0),1)
        # measure = err_real.data[0] + Variable(np.abs(balance) if GPU_NUMS > 0 else np.abs(balance))

        proBar.show(epoch, errD.item(), errG.item())

    noise = torch.randn(100, NOISE_DIM)
    noise_var = Variable(noise.cuda() if GPU_NUMS > 0 else noise)
    fake = Net_G(noise_var)
    torchvision.utils.save_image(fake.data,'outputs/Cifar10_%03d.png' % epoch, nrow=10)

