import torch
import torchvision

from torch.nn import Module, Conv2d, BatchNorm2d, Linear, UpsamplingNearest2d,\
    ReplicationPad2d, LeakyReLU, ReLU, Sigmoid, BCELoss
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

from lib.ProgressBar import ProgressBar
GPU_NUMS = 2
EPOCH = 30

IMAGE_CHANNEL = 3
IMAGE_SIZE = 64
NGF = 128
NDF = 128
LATENT_VARIABLE_SIZE = 50
BATCH_SIZE = 64

class VAE(Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.e1  = Conv2d(IMAGE_CHANNEL, NDF, 4, 2, 1)
        self.bn1 = BatchNorm2d(NDF)

        self.e2  = Conv2d(NDF, NDF*2, 4, 2, 1)
        self.bn2 = BatchNorm2d(NDF*2)

        self.e3  = Conv2d(NDF*2, NDF*4, 4, 2, 1)
        self.bn3 = BatchNorm2d(NDF*4)

        self.e4  = Conv2d(NDF*4, NDF*8, 4, 2, 1)
        self.bn4 = BatchNorm2d(NDF*8)

        self.e5  = Conv2d(NDF*8, NDF*8, 4, 2, 1)
        self.bn5 = BatchNorm2d(NDF*8)

        self.fc1 = Linear(NDF*8*4*4, LATENT_VARIABLE_SIZE)
        self.fc2 = Linear(NDF*8*4*4, LATENT_VARIABLE_SIZE)

        # decoder
        self.d1  = Linear(LATENT_VARIABLE_SIZE, NGF*8*2*4*4)

        self.up1 = UpsamplingNearest2d(scale_factor=2)
        self.pd1 = ReplicationPad2d(1)
        self.d2  = Conv2d(NGF*8*2, NGF*8, 3, 1)
        self.bn6 = BatchNorm2d(NGF*8, 1.e-3)

        self.up2 = UpsamplingNearest2d(scale_factor=2)
        self.pd2 = ReplicationPad2d(1)
        self.d3  = Conv2d(NGF*8, NGF*4, 3, 1)
        self.bn7 = BatchNorm2d(NGF*4, 1.e-3)

        self.up3 = UpsamplingNearest2d(scale_factor=2)
        self.pd3 = ReplicationPad2d(1)
        self.d4  = Conv2d(NGF*4, NGF*2, 3, 1)
        self.bn8 = BatchNorm2d(NGF*2, 1.e-3)

        self.up4 = UpsamplingNearest2d(scale_factor=2)
        self.pd4 = ReplicationPad2d(1)
        self.d5  = Conv2d(NGF*2, NGF, 3, 1)
        self.bn9 = BatchNorm2d(NGF, 1.e-3)

        self.up5 = UpsamplingNearest2d(scale_factor=2)
        self.pd5 = ReplicationPad2d(1)
        self.d6  = Conv2d(NGF, IMAGE_CHANNEL, 3, 1)

        self.leakyrelu = LeakyReLU(0.2)
        self.relu      = ReLU()
        self.sigmoid   = Sigmoid()
    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, NDF*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, NGF*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMAGE_CHANNEL, NDF, NGF))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if GPU_NUMS > 0:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

model = VAE()
if GPU_NUMS > 0 :
    model.cuda()

reconstruction_function = BCELoss()
reconstruction_function.size_average = False
optimizer = Adam(model.parameters(), lr=1e-4)

dataset = ImageFolder(root='/input/face/64_crop', transform=Compose([ToTensor()]))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

bar = ProgressBar(EPOCH, len(train_loader), "Loss:%.3f")

model.train()
train_loss = 0
for epoch in range(EPOCH):
    for ii, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]
        data = Variable(image.cuda() if GPU_NUMS > 0 else image)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        bar.show(loss.data[0] / mini_batch)

    model.eval()
    z = torch.randn(BATCH_SIZE, model.latent_variable_size)
    z = Variable(z.cuda() if GPU_NUMS > 0 else z, volatile=True)
    recon = model.decode(z)
    torchvision.utils.save_image(recon.data, 'output/Face64_%02d.png' % (epoch + 1))

torch.save(model.state_dict(), "output/VAE_64_Face_Pytorch_Generator.pth")


