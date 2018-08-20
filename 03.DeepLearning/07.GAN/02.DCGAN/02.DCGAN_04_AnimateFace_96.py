# coding=utf-8
import torch as t
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, ToTensor, Normalize,Compose,Resize
from torchvision.datasets import ImageFolder
from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU,Tanh, Conv2d,\
    LeakyReLU, Sigmoid, BCELoss,DataParallel
from torch.optim import Adam
from torch.autograd import Variable

from lib.ProgressBar import ProgressBar

CONFIG = {
    "EPOCHS" : 100,
    "NOISE_DIM" : 100,
    "NGF" : 64,
    "NDF" : 64,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 96,
    "IMAGE_CHANNEL" : 3,
    "BATCH_SIZE" : 256
}
class GeneratorNet(Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.mainNetwork = Sequential(
            # 100,1,1 => 64*8,4,4
            ConvTranspose2d(CONFIG["NOISE_DIM"], CONFIG["NGF"] * 8,
                            kernel_size=4, stride=1, padding=0, bias=False),
            BatchNorm2d(CONFIG["NGF"] * 8),
            ReLU(True),

            # 64*8,4,4 => 64*4,8,8
            ConvTranspose2d(CONFIG["NGF"] * 8, CONFIG["NGF"] * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(CONFIG["NGF"] * 4),
            ReLU(True),

            # 64*4,8,8 => 64*2,16,16
            ConvTranspose2d(CONFIG["NGF"] * 4, CONFIG["NGF"] * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(CONFIG["NGF"] * 2),
            ReLU(True),

            # 64*2,16,16 => 64,32,32
            ConvTranspose2d(CONFIG["NGF"] * 2, CONFIG["NGF"],
                            kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(CONFIG["NGF"]),
            ReLU(True),

            # 64*2,32,32 => 3,96,96
            ConvTranspose2d(CONFIG["NGF"], 3, kernel_size=5,
                            stride=3, padding=1, bias=False),
            Tanh() # 3 * 96 * 96
        )

    def forward(self, input):
        return self.mainNetwork(input)

class DiscriminatorNet(Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.mainNetwork = Sequential(
            # 3,96,96 => 64,32,32
            Conv2d(in_channels=CONFIG["IMAGE_CHANNEL"], out_channels=CONFIG["NDF"],
                   kernel_size=5, stride=3, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),

            #64,32,32 => 64*2,16,16
            Conv2d(in_channels=CONFIG["NDF"], out_channels=CONFIG["NDF"] * 2,
                   kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(CONFIG["NDF"] * 2),
            LeakyReLU(0.2, inplace=True),

            #64*2,16,16 => 64*4,8,8
            Conv2d(CONFIG["NDF"] * 2, CONFIG["NDF"] * 4,
                   kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(CONFIG["NDF"] * 4),
            LeakyReLU(0.2, inplace=True),

            #64*4,8,8 => 64*8,4,4
            Conv2d(CONFIG["NDF"] * 4, CONFIG["NDF"] * 8,
                   kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(CONFIG["NDF"] * 8),
            LeakyReLU(0.2, inplace=True),

            Conv2d(CONFIG["NDF"] * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            Sigmoid()
        )
    def forward(self, input):
        return self.mainNetwork(input).view(-1)

transforms = Compose([
    Resize(CONFIG["IMAGE_SIZE"]),
    CenterCrop(CONFIG["IMAGE_SIZE"]),
    ToTensor(),
    Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
dataset = ImageFolder('/input/AnimateFace', transform=transforms)
dataLoader = DataLoader(dataset=dataset, batch_size=CONFIG["BATCH_SIZE"],
                        shuffle=True,
                        drop_last=True)
netG, netD = DataParallel(GeneratorNet()), DataParallel(DiscriminatorNet())
map_location = lambda storage, loc: storage

optimizer_generator = Adam(netG.parameters(), 2e-4, betas=(0.5, 0.999))
optimizer_discriminator = Adam(netD.parameters(), 2e-4,betas=(0.5, 0.999))

criterion = BCELoss()

true_labels = Variable(t.ones(CONFIG["BATCH_SIZE"]))
fake_labels = Variable(t.zeros(CONFIG["BATCH_SIZE"]))
fix_noises = Variable(t.randn(CONFIG["BATCH_SIZE"],CONFIG["NOISE_DIM"],1,1))
noises = Variable(t.randn(CONFIG["BATCH_SIZE"],CONFIG["NOISE_DIM"],1,1))

if CONFIG["GPU_NUMS"] > 0:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    true_labels,fake_labels = true_labels.cuda(), fake_labels.cuda()
    fix_noises,noises = fix_noises.cuda(),noises.cuda()

proBar = ProgressBar(CONFIG["EPOCHS"], len(dataLoader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    if epoch % 30 == 0:
        optimizer_discriminator.param_groups[0]['lr'] /= 10
        optimizer_generator.param_groups[0]['lr'] /= 10

    for ii,(img,_) in enumerate(dataLoader):
        real_img = Variable(img.cuda() if CONFIG["GPU_NUMS"] > 1 else img)

        if ii % 1 ==0:
            # 训练判别器
            optimizer_discriminator.zero_grad()
            ## 尽可能的把真图片判别为正确
            output = netD(real_img)
            error_d_real = criterion(output,true_labels)
            error_d_real.backward()

            ## 尽可能把假图片判别为错误
            noises.data.copy_(t.randn(CONFIG["BATCH_SIZE"],CONFIG["NOISE_DIM"],1,1))
            fake_img = netG(noises).detach() # 根据噪声生成假图
            output = netD(fake_img)
            error_d_fake = criterion(output,fake_labels)
            error_d_fake.backward()
            optimizer_discriminator.step()

            error_d = error_d_fake + error_d_real

        if ii % 1==0:
            # 训练生成器
            optimizer_generator.zero_grad()
            noises.data.copy_(t.randn(CONFIG["BATCH_SIZE"],CONFIG["NOISE_DIM"],1,1))
            fake_img = netG(noises)
            output = netD(fake_img)
            error_g = criterion(output,true_labels)
            error_g.backward()
            optimizer_generator.step()

        proBar.show(epoch, error_d.item(), error_g.item())

    # 保存模型、图片
    fix_fake_imgs = netG(fix_noises)
    tv.utils.save_image(fix_fake_imgs.data[:64],'outputs/Pytorch_AnimateFace_%03d.png' %epoch,
                        normalize=True,range=(-1,1))

t.save(netG.state_dict(), "outputs/DCGAN_AnimateFace_Pytorch_Generator.pth")
