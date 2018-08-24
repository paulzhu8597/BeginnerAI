import torch
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.utils as vutils

import lib.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/Cat128/",
    "EPOCH" : 100,
    "GPU_NUMS" : 1,
    "BATCH_SIZE" : 64,
    "NOISE_DIM" : 100,
    "IMAGE_CHANNEL": 3,
    "IMAGE_SIZE" : 128,
    "LEARNING_RATE" : .0001
}

if CONFIG["GPU_NUMS"] > 0:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

trans = transf.Compose([
    transf.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
    transf.ToTensor(),
    transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

data = dset.ImageFolder(root=CONFIG["DATA_PATH"], transform=trans)

# Loading data in batch
dataset = torch.utils.data.DataLoader(data, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(CONFIG["NOISE_DIM"], 1024, 4, 1, 0, bias=False),
            # torch.nn.BatchNorm2d(1024),
            # torch.nn.ReLU()
            torch.nn.SELU(inplace=True)
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(512),
            # torch.nn.ReLU()
            torch.nn.SELU(inplace=True)
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.ReLU()
            torch.nn.SELU(inplace=True)
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.ReLU()
            torch.nn.SELU(inplace=True)
        )

        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(64),
            # torch.nn.ReLU(),
            torch.nn.SELU(inplace=True)
        )

        self.deconv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        output = self.deconv1(x)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        output = self.deconv6(output)
        return output

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(CONFIG["IMAGE_CHANNEL"], 64, 4, 2, 1, bias=False),
            # torch.nn.LeakyReLU(0.2, inplace=True)
            torch.nn.SELU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.LeakyReLU(0.2, inplace=True)
            torch.nn.SELU(inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.LeakyReLU(0.2, inplace=True)
            torch.nn.SELU(inplace=True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(512),
            # torch.nn.LeakyReLU(0.2, inplace=True)
            torch.nn.SELU(inplace=True)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            # torch.nn.BatchNorm2d(1024),
            # torch.nn.LeakyReLU(0.2, inplace=True)
            torch.nn.SELU(inplace=True)
        )

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
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

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

## Initialization
G = Generator()
D = Discriminator()

# Initialize weights
G.apply(weights_init)
D.apply(weights_init)

# Soon to be variables
x = torch.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["IMAGE_CHANNEL"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
z = torch.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["NOISE_DIM"], 1, 1)
# This is to see during training, size and values won't change
z_test = torch.FloatTensor(100, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)

if CONFIG["GPU_NUMS"] > 0:
    G = G.cuda()
    D = D.cuda()
    x = x.cuda()
    z = z.cuda()
    z_test = z_test.cuda()

x = Variable(x)
z = Variable(z)
z_test = Variable(z_test)

optimizerD = torch.optim.Adam(D.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999), weight_decay=0)
optimizerG = torch.optim.Adam(G.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999), weight_decay=0)

## Fitting model
bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(dataset), "D loss:%.3f;G loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):

    if epoch == 30:
        optimizerD.param_groups[0]['lr'] /= 2
        optimizerG.param_groups[0]['lr'] /= 2

    for i, data_batch in enumerate(dataset, 0):

        for p in D.parameters():
            p.requires_grad = True

        # Train with real data
        D.zero_grad()
        # We can ignore labels since they are all cats!
        images, labels = data_batch
        # Mostly necessary for the last one because if N might not be a multiple of batch_size
        current_batch_size = images.size(0)
        if CONFIG["GPU_NUMS"] > 0:
            images = images.cuda()
        # Transfer batch of images to x
        x.data.resize_as_(images).copy_(images)
        y_pred = D(x)
        errD_real = 0.5 * torch.mean((y_pred - 1) ** 2)
        errD_real.backward()

        # Train with fake data
        z.data.resize_(current_batch_size, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
        x_fake = G(z)
        # Detach y_pred from the neural network G and put it inside D
        y_pred_fake = D(x_fake.detach())
        errD_fake = 0.5 * torch.mean((y_pred_fake - 0) ** 2)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ########################
        # (2) Update G network #
        ########################

        # Make it a tiny bit faster
        for p in D.parameters():
            p.requires_grad = False

        G.zero_grad()
        y_pred_fake = D(x_fake)
        errG = 0.5 * torch.mean((y_pred_fake - 1) ** 2)
        errG.backward(retain_graph=True)
        optimizerG.step()

        current_step = i + epoch*len(dataset)
        # Log results so we can see them in TensorBoard after

        bar.show(epoch, errD.item(), errG.item())

    fake_test = G(z_test)
    vutils.save_image(fake_test.data, 'outputs/Cat_%03d.png' % (epoch), nrow=10, normalize=True)