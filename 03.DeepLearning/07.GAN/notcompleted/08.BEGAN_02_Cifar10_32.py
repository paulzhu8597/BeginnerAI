import torch as t
import torchvision as tv

import torch.autograd as t_auto
import torch.optim as t_optim

import lib.ProgressBar as j_bar
import lib.dataset.pytorch_dataset as j_data

CONFIG = {
    ""
}

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3
NOISE_DIM = 64
GPU_NUMS = 0
BATCH_SIZE = 128
EPOCHS = 100

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_encoder = t.nn.Sequential(
            t.nn.Conv2d(3, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64, 64 * 2, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.MaxPool2d(2, 2),
            t.nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64 * 2, 64 * 3, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.MaxPool2d(2, 2),
            t.nn.Conv2d(64 * 3, 64 * 3, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64 * 3, 64 * 3, 3, 1, 1),
            t.nn.ELU(inplace=True),
            # t.nn.MaxPool2d(2, 2),
            # t.nn.Conv2d(64 * 3, 64 * 3, 3, 1, 1),
            # t.nn.ELU(inplace=True),
            # t.nn.Conv2d(64 * 3, 64 * 3, 3, 1, 1),
            # t.nn.ELU(inplace=True)
        )

        self.fc_encoder = t.nn.Sequential(
            # state size. (n_hidden*8) x 8 x 8
            t.nn.Linear(8 * 8 * 3 * 64, NOISE_DIM)
            # input is Z, going into a convolution
        )

        self.fc_decoder = t.nn.Sequential(
            # input is Z, going into a convolution
            t.nn.Linear(NOISE_DIM, 8 * 8 * 64)
            # state size. (n_hidden) x 8 x 8
        )

        self.conv_decoder = t.nn.Sequential(
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Upsample(scale_factor=2),
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Upsample(scale_factor=2),
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True),
            t.nn.Conv2d(64, IMAGE_CHANNEL, 3, 1, 1),
        )
    def forward(self, x):
        h = self.conv_encoder(x)
        h = h.view(-1, 8 * 8 * 3 * 64)
        h = self.fc_encoder(h)
        h = self.fc_decoder(h)
        h = h.view(-1, 64, 8, 8)
        return self.conv_decoder(h)

class Generator(t.nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(NOISE_DIM, 64 * 8 * 8)
        )

        self.conv = t.nn.Sequential(
            t.nn.Conv2d(64, 64, 3, 1, 1),
            t.nn.ELU(inplace=True)
        )

        self.up1 = t.nn.Upsample(scale_factor=2)

        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(64, IMAGE_CHANNEL, 3, 1, 1)
        )

    def forward(self,x):
        output = self.fc1(x)
        output = output.view(-1, 64, 8, 8)
        output = self.conv(output)
        output = self.conv(output)
        output = self.up1(output)

        output = self.conv(output)
        output = self.conv(output)
        output = self.up1(output)

        output = self.conv(output)
        output = self.conv(output)

        return output

Net_D=Discriminator()
Net_G = Generator()

if GPU_NUMS > 0:
    Net_D.cuda()
    Net_G.cuda()

###########   LOSS & OPTIMIZER   ##########
optimizerD = t_optim.Adam(Net_D.parameters(),lr=0.0001, betas=(0.5, 0.999))
optimizerG = t_optim.Adam(Net_G.parameters(),lr=0.0001, betas=(0.5, 0.999))

dataset = j_data.Cifar10DataSetForPytorch(train=True, transform=tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        # Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ]))
train_loader = t.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

noise = t.randn(100, NOISE_DIM)
noise_var =  t_auto.Variable(noise.cuda() if GPU_NUMS > 0 else noise)

k = 0
proBar = j_bar.ProgressBar(EPOCHS, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    for index, (images,_) in enumerate(train_loader):
        mini_batch = images.shape[0]
        noise = t_auto.Variable(t.FloatTensor(mini_batch, NOISE_DIM, 1, 1).cuda() if GPU_NUMS > 0 else t.FloatTensor(mini_batch, NOISE_DIM, 1, 1))
        real =  t_auto.Variable(t.FloatTensor(mini_batch, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE).cuda() if GPU_NUMS > 0 else t.FloatTensor(mini_batch, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        label =  t_auto.Variable(t.FloatTensor(1).cuda() if GPU_NUMS > 0 else t.FloatTensor(1))

        Net_D.zero_grad()
        real.data.resize_(images.size()).copy_(images)

        # generate fake data
        noise.data.resize_(images.size(0), NOISE_DIM)
        noise.data.uniform_(-1,1)
        fake = Net_G(noise)

        fake_recons = Net_D(fake.detach())
        real_recons = Net_D(real)

        err_real = t.mean(t.abs(real_recons-real))
        err_fake = t.mean(t.abs(fake_recons-fake))

        errD = err_real - k*err_fake
        errD.backward()
        optimizerD.step()

        Net_G.zero_grad()
        fake = Net_G(noise)
        fake_recons = Net_D(fake)
        errG = t.mean(t.abs(fake_recons-fake))
        errG.backward()
        optimizerG.step()

        balance = (0.5 * err_real - err_fake).data[0]
        k = min(max(k + 0.001 * balance,0),1)
        # measure = err_real.data[0] + Variable(np.abs(balance) if GPU_NUMS > 0 else np.abs(balance))

        proBar.show(epoch, errD.item(), errG.item())

    fake = Net_G(noise_var)
    tv.utils.save_image(fake.data,'outputs/Cifar10_%03d.png' % epoch, nrow=10)

