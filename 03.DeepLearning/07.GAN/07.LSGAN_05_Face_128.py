import torch as t
import torchvision as tv

import torch.utils.data as t_data
import torch.autograd as t_auto
import torch.optim as t_optim

import lib.dataset.pytorch_dataset as j_data
import lib.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/Faces/SquareImages/",
    "EPOCH" : 100,
    "GPU_NUMS" : 1,
    "BATCH_SIZE" : 64,
    "NOISE_DIM" : 62,
    "IMAGE_CHANNEL": 3,
    "IMAGE_SIZE" : 128,
    "LEARNING_RATE" : 2e-4
}

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, t.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, t.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, t.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(CONFIG["NOISE_DIM"], 1024 * 4 * 4),
            t.nn.BatchNorm1d(1024 * 4 * 4),
            t.nn.ReLU()
        )

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(1024, 512, 2, 2, bias=False),
            t.nn.BatchNorm2d(512),
            t.nn.ReLU()
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(512, 256, 2, 2, bias=False),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU()
        )

        self.deconv3 = t.nn.Sequential(
            t.nn.ConvTranspose2d(256, 128, 2, 2, bias=False),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU()
        )

        self.deconv4 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 64, 2, 2, bias=False),
        )

        self.deconv5 = t.nn.Sequential(
            t.nn.ConvTranspose2d(64, CONFIG["IMAGE_CHANNEL"], 2, 2, bias=False),
            t.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        output = self.fc1(x)
        output = output.view(-1, 1024, 4, 4)
        output = self.deconv1(output)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        return output

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(CONFIG["IMAGE_CHANNEL"], 64, 2, 2, bias=False),
            t.nn.LeakyReLU(0.2)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 2, 2, bias=False),
            t.nn.BatchNorm2d(128),
            t.nn.LeakyReLU(0.2)
        )

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 2, 2, bias=False),
            t.nn.BatchNorm2d(256),
            t.nn.LeakyReLU(0.2)
        )

        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(256, 512, 2, 2, bias=False),
            t.nn.BatchNorm2d(512),
            t.nn.LeakyReLU(0.2)
        )

        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(512, 1024, 2, 2, bias=False),
            t.nn.BatchNorm2d(1024),
            t.nn.LeakyReLU(0.2)
        )

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(1024 * 4 * 4, 1024),
            t.nn.BatchNorm1d(1024),
            t.nn.LeakyReLU(0.2)
        )

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(1024, 1),
            t.nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.view(-1, 1024 * 4 * 4)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

NetD = Discriminator()
NetG = Generator()
MSE_LOSS = t.nn.MSELoss()
if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    MSE_LOSS = MSE_LOSS.cuda()

optimizerD = t_optim.Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))
optimizerG = t_optim.Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))

dataset = tv.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=tv.transforms.Compose([
    tv.transforms.Resize(CONFIG["IMAGE_SIZE"]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = t_data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

fix_noise = t.randn(100, CONFIG["NOISE_DIM"])
fix_noise_var = t_auto.Variable(fix_noise.cuda() if CONFIG["GPU_NUMS"] > 0 else fix_noise)

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "D Loss:%.3f;G Loss:%.3f")
NetD.train()
for epoch in range(1, CONFIG["EPOCH"] + 1):
    NetG.train()
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]

        noise = t.rand(mini_batch, CONFIG["NOISE_DIM"])

        real_var = t_auto.Variable(image.cuda() if CONFIG["GPU_NUMS"] > 0 else image)
        noise_var = t_auto.Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)

        label_real_var = t_auto.Variable(t.ones(mini_batch, 1).cuda()  if CONFIG["GPU_NUMS"] > 0 else t.ones(mini_batch, 1))
        label_fake_var = t_auto.Variable(t.zeros(mini_batch, 1).cuda() if CONFIG["GPU_NUMS"] > 0 else t.zeros(mini_batch, 1))

        optimizerD.zero_grad()

        D_real = NetD(real_var)
        D_real_loss = MSE_LOSS(D_real, label_real_var)

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = MSE_LOSS(D_fake, label_fake_var)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizerD.step()

        optimizerG.zero_grad()

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        G_loss = MSE_LOSS(D_fake, label_real_var)

        G_loss.backward()
        optimizerG.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/Face_%03d.png' % epoch,nrow=10)

