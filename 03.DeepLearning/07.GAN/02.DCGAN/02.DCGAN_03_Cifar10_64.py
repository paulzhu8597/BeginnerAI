# coding=utf-8
import torch
import torchvision as tv

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms

from lib.dataset.pytorch_dataset import Cifar10DataSetForPytorch
from lib.ProgressBar import ProgressBar

CONFIG = {
    "LEARNING_RATE" : 2e-4,
    "IMAGE_CHANNEL" : 3,
    "IMAGE_SIZE" : 64,
    "NGF" : 64,
    "NDF" : 64,
    "BETA1" : 0.5,
    "BATCH_SIZE" : 64,
    "EPOCHS" : 100,
    "GPU_NUMS" : 0,
    "NOISE_DIM" : 100
}

transform=transforms.Compose([
    transforms.Resize(CONFIG["IMAGE_SIZE"]) ,
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = Cifar10DataSetForPytorch(train=True, transform=transform)
dataloader=torch.utils.data.DataLoader(dataset,CONFIG["BATCH_SIZE"],shuffle = True)

netg = nn.Sequential(
    nn.ConvTranspose2d(CONFIG["NOISE_DIM"],CONFIG["NGF"]*8,4,1,0,bias=False),
    nn.BatchNorm2d(CONFIG["NGF"]*8),
    nn.ReLU(True),

    nn.ConvTranspose2d(CONFIG["NGF"]*8,CONFIG["NGF"]*4,4,2,1,bias=False),
    nn.BatchNorm2d(CONFIG["NGF"]*4),
    nn.ReLU(True),

    nn.ConvTranspose2d(CONFIG["NGF"]*4,CONFIG["NGF"]*2,4,2,1,bias=False),
    nn.BatchNorm2d(CONFIG["NGF"]*2),
    nn.ReLU(True),

    nn.ConvTranspose2d(CONFIG["NGF"]*2,CONFIG["NGF"],4,2,1,bias=False),
    nn.BatchNorm2d(CONFIG["NGF"]),
    nn.ReLU(True),

    nn.ConvTranspose2d(CONFIG["NGF"],CONFIG["IMAGE_CHANNEL"],4,2,1,bias=False),
    nn.Tanh()
)

netd = nn.Sequential(
    nn.Conv2d(CONFIG["IMAGE_CHANNEL"],CONFIG["NDF"],4,2,1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(CONFIG["NDF"],CONFIG["NDF"]*2,4,2,1,bias=False),
    nn.BatchNorm2d(CONFIG["NDF"]*2),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(CONFIG["NDF"]*2,CONFIG["NDF"]*4,4,2,1,bias=False),
    nn.BatchNorm2d(CONFIG["NDF"]*4),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(CONFIG["NDF"]*4,CONFIG["NDF"]*8,4,2,1,bias=False),
    nn.BatchNorm2d(CONFIG["NDF"]*8),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(CONFIG["NDF"]*8,1,4,1,0,bias=False),
    nn.Sigmoid()
)

if CONFIG["GPU_NUMS"] > 1 :
    netd = nn.DataParallel(netd)
    netg = nn.DataParallel(netg)

optimizerD = Adam(netd.parameters(),lr=CONFIG["LEARNING_RATE"],betas=(CONFIG["BETA1"],0.999))
optimizerG = Adam(netg.parameters(),lr=CONFIG["LEARNING_RATE"],betas=(CONFIG["BETA1"],0.999))

# criterion
criterion = nn.BCELoss()

fix_noise = Variable(torch.FloatTensor(CONFIG["BATCH_SIZE"],CONFIG["NOISE_DIM"],1,1).normal_(0,1))
if CONFIG["GPU_NUMS"] > 0:
    fix_noise = fix_noise.cuda()
    netd.cuda()
    netg.cuda()
    criterion.cuda() # it's a good habit

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

        netd.zero_grad()
        output=netd(input)
        error_real=criterion(output.squeeze(),label)
        error_real.backward()

        D_x=output.data.mean()
        fake_pic=netg(noise).detach()
        output2=netd(fake_pic)
        label.data.fill_(0) # 0 for fake
        error_fake=criterion(output2.squeeze(),label)

        error_fake.backward()
        D_x2=output2.data.mean()
        error_D=error_real+error_fake
        optimizerD.step()

        netg.zero_grad()
        label.data.fill_(1)
        noise.data.normal_(0,1)
        fake_pic=netg(noise)
        output=netd(fake_pic)
        error_G=criterion(output.squeeze(),label)
        error_G.backward()

        optimizerG.step()
        D_G_z2=output.data.mean()
        bar.show(epoch, error_D.item(), error_G.item())

    fake_u=netg(fix_noise)

    tv.utils.save_image(fake_u.data[:64], "outputs/Cifar10_%03d.png" % epoch,normalize=True,range=(-1,1))

torch.save(netg.state_dict(), "outputs/NetG_Cifar.pth")