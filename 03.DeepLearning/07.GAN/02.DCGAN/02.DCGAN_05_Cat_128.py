# coding=utf-8
import torch as t
import torchvision as tv
import torchvision.datasets as dset
from lib.ProgressBar import ProgressBar

CONFIG = {
    "DATA_PATH" : "/input/Cat128/",
    "LEARNING_RATE_D" : .00005,
    "LEARNING_RATE_G" : .0002,
    "IMAGE_CHANNEL" : 3,
    "IMAGE_SIZE" : 128,
    "BETA1" : 0.5,
    "BATCH_SIZE" : 64,
    "EPOCHS" : 100,
    "GPU_NUMS" : 1,
    "NOISE_DIM" : 100
}

class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(CONFIG["NOISE_DIM"], 1024, 4, 1, 0, bias=False),
            # t.nn.BatchNorm2d(1024),
            # t.nn.ReLU()
            t.nn.SELU(inplace=True)
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(512),
            # t.nn.ReLU()
            t.nn.SELU(inplace=True)
        )

        self.deconv3 = t.nn.Sequential(
            t.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(256),
            # t.nn.ReLU()
            t.nn.SELU(inplace=True)
        )

        self.deconv4 = t.nn.Sequential(
            t.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(128),
            # t.nn.ReLU()
            t.nn.SELU(inplace=True)
        )

        self.deconv5 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(64),
            # t.nn.ReLU()
            t.nn.SELU(inplace=True)
        )

        self.deconv6 = t.nn.Sequential(
            t.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            t.nn.Tanh()
        )

    def forward(self, x):
        output = self.deconv1(x)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        output = self.deconv6(output)
        return output

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            # t.nn.LeakyReLU(0.2, inplace=True)
            t.nn.SELU(inplace=True)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(128),
            # t.nn.LeakyReLU(0.2, inplace=True)
            t.nn.SELU(inplace=True)
        )

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(256),
            # t.nn.LeakyReLU(0.2, inplace=True)
            t.nn.SELU(inplace=True)
        )

        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(512),
            # t.nn.LeakyReLU(0.2, inplace=True)
            t.nn.SELU(inplace=True)
        )

        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            # t.nn.BatchNorm2d(1024),
            # t.nn.LeakyReLU(0.2, inplace=True)
            t.nn.SELU(inplace=True)
        )

        self.conv6 = t.nn.Sequential(
            t.nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(-1)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

NetD = Discriminator()
NetG = Generator()
NetG.apply(weights_init)
NetD.apply(weights_init)
criterion = t.nn.BCELoss()

x = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["IMAGE_CHANNEL"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
y = t.FloatTensor(CONFIG["BATCH_SIZE"])
z = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["NOISE_DIM"], 1, 1)
z_test = t.FloatTensor(100, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    criterion = criterion.cuda()
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()
    z_test = z_test.cuda()

optimizerD = t.optim.Adam(NetD.parameters(),lr=CONFIG["LEARNING_RATE_D"],betas=(CONFIG["BETA1"],0.999), weight_decay=0)
optimizerG = t.optim.Adam(NetG.parameters(),lr=CONFIG["LEARNING_RATE_G"],betas=(CONFIG["BETA1"],0.999), weight_decay=0)

transform=tv.transforms.Compose([
    tv.transforms.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])) ,
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = dset.ImageFolder(root=CONFIG["DATA_PATH"], transform=transform)
dataloader = t.utils.data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

bar = ProgressBar(CONFIG["EPOCHS"], len(dataloader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    for i, data_batch in enumerate(dataloader, 0):
        for p in NetD.parameters():
            p.requires_grad = True

        NetD.zero_grad()
        images, labels = data_batch
        current_batch_size = images.size(0)
        images = images.cuda() if CONFIG["GPU_NUMS"] > 0 else images
        x.data.resize_as_(images).copy_(images)
        y.data.resize_(current_batch_size).fill_(1)
        y_pred = NetD(x)
        errD_real = criterion(y_pred, y)
        errD_real.backward()
        D_real = y_pred.data.mean()

        z.data.resize_(current_batch_size,CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
        x_fake = NetG(z)
        y.data.resize_(current_batch_size).fill_(0)
        y_pred_fake = NetD(x_fake.detach())
        errD_fake = criterion(y_pred_fake, y)
        errD_fake.backward()
        D_fake = y_pred_fake.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        for p in NetD.parameters():
            p.requires_grad = False

        NetG.zero_grad()
        y.data.resize_(current_batch_size).fill_(1)
        y_pred_fake = NetD(x_fake)
        errG = criterion(y_pred_fake, y)
        errG.backward(retain_graph=True)
        D_G = y_pred_fake.data.mean()
        optimizerG.step()

        bar.show(epoch, errD.item(), errG.item())
    fake_test = NetG(z_test)
    tv.utils.save_image(fake_test.data, 'outputs/Cat_%03d.png' %epoch, nrow=10, normalize=True)
