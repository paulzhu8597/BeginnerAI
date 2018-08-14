import torch
import torchvision

from torch.optim import Adam
from torchvision.transforms import ToTensor, Compose

from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU, ConvTranspose2d, BatchNorm2d, Tanh, Conv2d, LeakyReLU,\
    Sigmoid, BCELoss
from torch.autograd import Variable

from lib.dataset.pytorch_dataset import MNISTDataSetForPytorch
from lib.ProgressBar import ProgressBar

CONFIG = {
    "DATA_PATH" : "/input/mnist.npz",
    "EPOCHS" : 100,
    "BATCH_SIZE" : 128,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 28,
    "IMAGE_CHANNEL" : 1,
    "NOISE_DIM" : 100,
    "LEARNING_RATE" : 2e-4,
    "BETA1" : 0.5
}

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = Sequential(
            Linear(CONFIG["NOISE_DIM"], 1024),
            BatchNorm1d(num_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=128 * 7 * 7),
            BatchNorm1d(num_features=128*7*7),
            ReLU()
        )

        self.deconv = Sequential(
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(num_features=64),
            ReLU(),
            ConvTranspose2d(in_channels=64, out_channels=CONFIG["IMAGE_CHANNEL"], kernel_size=4, stride=2, padding=1),
            Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        output = x.view(-1,CONFIG["NOISE_DIM"])
        output = self.fc(output)
        output = output.view(-1, 128, 7, 7)
        output =self.deconv(output)

        return output

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels=CONFIG["IMAGE_CHANNEL"], out_channels=64, kernel_size=4, stride=2, padding=1),
            LeakyReLU(negative_slope=0.2),
            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(num_features=128),
            LeakyReLU(negative_slope=0.2)
        )

        self.fc = Sequential(
            Linear(128 * 7 * 7, 1024),
            BatchNorm1d(1024),
            LeakyReLU(negative_slope=0.2),
            Linear(1024, 1),
            Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        output = self.conv(x)
        output = output.view(-1, 128 * 7 * 7)
        output = self.fc(output)

        return output

NetG = Generator()
NetD = Discriminator()

optimizerD = Adam(NetD.parameters(),lr=CONFIG["LEARNING_RATE"],betas=(CONFIG["BETA1"],0.999))
optimizerG = Adam(NetG.parameters(),lr=CONFIG["LEARNING_RATE"],betas=(CONFIG["BETA1"],0.999))
criterion = BCELoss()

fix_noise = Variable(torch.FloatTensor(CONFIG["BATCH_SIZE"],CONFIG["NOISE_DIM"],1,1).normal_(0,1))
if CONFIG["GPU_NUMS"] > 0:
    NetD = NetD.cuda()
    NetG = NetG.cuda()
    fix_noise = fix_noise.cuda()
    criterion.cuda() # it's a good habit

transform=Compose([
    ToTensor()
])

dataset = MNISTDataSetForPytorch(root=CONFIG["DATA_PATH"],train=True, transform=transform)
dataloader=torch.utils.data.DataLoader(dataset,CONFIG["BATCH_SIZE"],shuffle = True)

bar = ProgressBar(CONFIG["EPOCHS"], len(dataloader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    if epoch % 30 == 0:
        optimizerD.param_groups[0]['lr'] /= 10
        optimizerG.param_groups[0]['lr'] /= 10

    for ii, data in enumerate(dataloader,0):
        real,_=data
        input = Variable(real.cuda() if CONFIG["GPU_NUMS"] > 0 else real)
        label = torch.ones(input.size(0))
        label = Variable(label.cuda() if CONFIG["GPU_NUMS"] > 0 else label) # 1 for real
        noise = torch.randn(input.size(0),CONFIG["NOISE_DIM"],1,1)
        noise = Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)

        NetD.zero_grad()
        output=NetD(input)
        error_real=criterion(output.squeeze(),label)
        error_real.backward()

        D_x=output.data.mean()
        fake_pic=NetG(noise).detach()
        output2=NetD(fake_pic)
        label.data.fill_(0) # 0 for fake
        error_fake=criterion(output2.squeeze(),label)

        error_fake.backward()
        D_x2=output2.data.mean()
        error_D=error_real+error_fake
        optimizerD.step()

        NetG.zero_grad()
        label.data.fill_(1)
        noise.data.normal_(0,1)
        fake_pic=NetG(noise)
        output=NetD(fake_pic)
        error_G=criterion(output.squeeze(),label)
        error_G.backward()

        optimizerG.step()
        D_G_z2=output.data.mean()
        bar.show(epoch, error_D.item(), error_G.item())

    fake_u=NetG(fix_noise)

    torchvision.utils.save_image(fake_u.data[:64], "outputs/MNIST_%03d.png" % epoch,normalize=True,range=(-1,1))