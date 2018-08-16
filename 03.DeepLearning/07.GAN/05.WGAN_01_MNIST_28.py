import torch as t
import torch.utils.data as t_data
import torch.autograd as t_auto
import torch.optim as t_optim

import torchvision as tv

import lib.dataset.pytorch_dataset as j_data
import lib.ProgressBar as j_bar

CONFIG = {
    "NOISE_DIM" : 100,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 28,
    "IMAGE_CHANNEL" : 1,
    "BATCH_SIZE" : 64,
    "EPOCH" : 100,
    "CLAMP_NUM" : 0.01,
    "LEARNING_RATE" : 5e-5
}

class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(CONFIG["NOISE_DIM"], 1024),
            t.nn.BatchNorm1d(num_features=1024),
            t.nn.ReLU(inplace=True)
        )

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(1024, 128 * 7 * 7),
            t.nn.BatchNorm1d(num_features=128*7*7),
            t.nn.ReLU(inplace=True)
        )

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            t.nn.BatchNorm2d(num_features=64),
            t.nn.ReLU(inplace=True)
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(64, CONFIG["IMAGE_CHANNEL"], 4, 2, 1),
            t.nn.Tanh()
        )

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = output.view(-1, 128, 7, 7)
        output = self.deconv1(output)
        output = self.deconv2(output)

        return output

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(CONFIG["IMAGE_CHANNEL"], 64, 4, 2, 1),
            t.nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 4, 2, 1),
            t.nn.BatchNorm2d(num_features=128),
            t.nn.LeakyReLU(negative_slope=0.2)
        )

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(128*7*7, 1024),
            t.nn.BatchNorm1d(1024),
            t.nn.LeakyReLU(negative_slope=0.2)
        )

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(1024, 1)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.view(-1, 128*7*7)
        output = self.fc1(output)
        output = self.fc2(output)
        output = output.mean(0).view(1)

        return output

def weight_init(net):
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

NetD = Discriminator()
NetG = Generator()
NetG.apply(weight_init)
NetD.apply(weight_init)

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()

optimizerD = t_optim.RMSprop(NetD.parameters(), lr=CONFIG["LEARNING_RATE"])
optimizerG = t_optim.RMSprop(NetG.parameters(), lr=CONFIG["LEARNING_RATE"])

dataset = j_data.MNISTDataSetForPytorch(transform=tv.transforms.Compose([
    tv.transforms.Resize(CONFIG["IMAGE_SIZE"]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = t_data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

one=t.FloatTensor([1])
mone = -1 * one
one_var = t_auto.Variable(one.cuda() if CONFIG["GPU_NUMS"] > 0 else one)
mone_var = t_auto.Variable(mone.cuda() if CONFIG["GPU_NUMS"] > 0 else mone)

fix_noise = t.FloatTensor(100, CONFIG["NOISE_DIM"]).normal_(0,1)
fix_noise_var = t_auto.Variable(fix_noise.cuda() if CONFIG["GPU_NUMS"] > 0 else fix_noise)
bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    for index, (image, label) in enumerate(train_loader):
        real  = image
        real_var = t_auto.Variable(real.cuda() if CONFIG["GPU_NUMS"] > 0 else real)
        noise = t.randn(real_var.size(0),CONFIG["NOISE_DIM"])
        noise_var = t_auto.Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)

        for parm in NetD.parameters():
            parm.data.clamp_(-CONFIG["CLAMP_NUM"], CONFIG["CLAMP_NUM"])

        NetD.zero_grad()
        D_real=NetD(real_var)
        D_real.backward(one_var)

        fake_pic=NetG(noise_var).detach()
        D_fake=NetD(fake_pic)
        D_fake.backward(mone_var)
        optimizerD.step()

        G_ = D_fake
        if (index+1)%5 ==0:
            NetG.zero_grad()
            noise.data.normal_(0,1)
            fake_pic=NetG(noise_var)
            G_=NetD(fake_pic)
            G_.backward(one_var)
            optimizerG.step()
            if index%100==0:
                pass
        bar.show(epoch, D_fake.item(), G_.item())

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/mnist_%03d.png' % epoch,nrow=10)

