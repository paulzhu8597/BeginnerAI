import torch as t
import torchvision as tv

import lib.dataset.pytorch_dataset as j_data
import lib.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/AnimateFace/",
    "NOISE_DIM" : 100,
    "GPU_NUMS" : 0,
    "IMAGE_SIZE" : 96,
    "IMAGE_CHANNEL" : 3,
    "BATCH_SIZE" : 64,
    "EPOCH" : 100,
    "CLAMP_NUM" : 1e-2,
    "LEARNING_RATE" : 5e-4
}

def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)

class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=CONFIG["NOISE_DIM"], out_channels=64 * 8, kernel_size=4, stride=1, padding=0, bias=False),
            t.nn.BatchNorm2d(num_features=64 * 8),
            t.nn.ReLU(inplace=True)
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=64 * 8, out_channels=64 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            t.nn.BatchNorm2d(num_features=64 * 2),
            t.nn.ReLU(inplace=True)
        )

        self.deconv3 = t.nn.t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=64 * 4, out_channels=64 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            t.nn.BatchNorm2d(num_features=64 * 1),
            t.nn.ReLU(True)
        )

        self.deconv4 = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            t.nn.BatchNorm2d(num_features=64),
            t.nn.ReLU(True)
        )

        self.deconv5 = t.nn.t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=64, out_channels=CONFIG["IMAGE_CHANNEL"], kernel_size=4, stride=3, padding=1, bias=False),
            t.nn.Tanh()
        )

    def forward(self, x):
        network = self.deconv1(x)
        network = self.deconv2(network)
        network = self.deconv3(network)
        network = self.deconv4(network)
        network = self.deconv5(network)

        return network

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=CONFIG["IMAGE_CHANNEL"], out_channels=64, kernel_size=4, stride=3, padding=1, bias=False),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            t.nn.BatchNorm2d(num_features=64*2),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=64*2, out_channels=64*4,kernel_size=4, stride=2, padding=1, bias=False),
            t.nn.BatchNorm2d(num_features=64*4),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=64*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        network = self.conv1(x)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)
        network = self.conv5(network)
        network = network.mean(0).view(1)
        return network

NetD = Discriminator()
NetG = Generator()
NetD.apply(weight_init)
NetG.apply(weight_init)
if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()

optimizerD = t.optim.RMSprop(NetD.parameters(), lr=CONFIG["LEARNING_RATE"])
optimizerG = t.optim.RMSprop(NetG.parameters(), lr=CONFIG["LEARNING_RATE"])

dataset = tv.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=tv.transforms.Compose([
    tv.transforms.Resize(CONFIG["IMAGE_SIZE"]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5]*3, [0.5]*3)
]))
train_loader = t.utils.data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

one=t.FloatTensor([1])
mone = -1 * one

one_var =  t.autograd.Variable(one.cuda() if CONFIG["GPU_NUMS"] > 0 else one)
mone_var = t.autograd.Variable(mone.cuda() if CONFIG["GPU_NUMS"] > 0 else mone)

fix_noise = t.FloatTensor(100, CONFIG["NOISE_DIM"], 1, 1).normal_(0,1)
fix_noise_var = t.autograd.Variable(fix_noise.cuda() if CONFIG["GPU_NUMS"] > 0 else fix_noise)

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(CONFIG["EPOCH"]):
    for index, (image, label) in enumerate(train_loader):
        real  = image
        real_var = t.autograd.Variable(real.cuda() if CONFIG["GPU_NUMS"] > 0 else real)
        noise = t.randn(real_var.size(0),CONFIG["NOISE_DIM"],1,1)
        noise_var = t.autograd.Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)

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
        bar.show(D_fake.data[0], G_.data[0])

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/Faces_%03d.png' % epoch,nrow=10)