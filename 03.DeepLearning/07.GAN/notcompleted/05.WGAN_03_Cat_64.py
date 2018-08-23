import torch as t
import torchvision as tv
import lib.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/Cat64/",
    "NOISE_DIM" : 100,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 64,
    "IMAGE_CHANNEL" : 3,
    "BATCH_SIZE" : 64,
    "EPOCH" : 100,
    "CLAMP_NUM" : 1e-2,
    "LEARNING_RATE" : .00005
}

class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(CONFIG["NOISE_DIM"], 1024, 4, 1, 0, bias=False),
            t.nn.BatchNorm2d(1024),
            t.nn.ReLU()
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            t.nn.BatchNorm2d(512),
            t.nn.ReLU()
        )

        self.deconv3 = t.nn.Sequential(
            t.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU()
        )

        self.deconv4 = t.nn.Sequential(
            t.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU()
        )

        self.deconv5 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            t.nn.Tanh()
        )

    def forward(self, x):
        output = self.deconv1(x)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)
        return output

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            t.nn.LeakyReLU(0.2, inplace=True)
            # torch.nn.SELU(inplace=True)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            t.nn.BatchNorm2d(256),
            t.nn.LeakyReLU(0.2, inplace=True)
            # torch.nn.SELU(inplace=True)
        )

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            t.nn.BatchNorm2d(512),
            t.nn.LeakyReLU(0.2, inplace=True)
            # torch.nn.SELU(inplace=True)
        )

        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            t.nn.BatchNorm2d(1024),
            t.nn.LeakyReLU(0.2, inplace=True)
            # torch.nn.SELU(inplace=True)
        )

        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.mean(0)
        output = output.view(1)

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

NetG = Generator()
NetD = Discriminator()
NetG.apply(weights_init)
NetD.apply(weights_init)



trans = tv.transforms.Compose([
    tv.transforms.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

data = tv.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=trans)
dataset = t.utils.data.DataLoader(data, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

x = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["IMAGE_CHANNEL"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
z = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["NOISE_DIM"], 1, 1)
z_test = t.FloatTensor(100, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
one = t.FloatTensor([1])
one_neg = one * -1

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    x = x.cuda()
    z = z.cuda()
    z_test = z_test.cuda()
    one, one_neg = one.cuda(), one_neg.cuda()

x = t.autograd.Variable(x)
z = t.autograd.Variable(z)
z_test = t.autograd.Variable(z_test)

optimizerD = t.optim.RMSprop(NetD.parameters(), lr=CONFIG["LEARNING_RATE"])
optimizerG = t.optim.RMSprop(NetG.parameters(), lr=CONFIG["LEARNING_RATE"])

gen_iterations = 0
bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(dataset), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    i = 0
    data_iter = iter(dataset)

    while i < len(dataset):

        for p in NetD.parameters():
            p.requires_grad = True

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            N_critic = 100
        else:
            N_critic = 5

        t = 0
        while t < N_critic and i < len(dataset):

            NetD.zero_grad()

            for p in NetD.parameters():
                p.data.clamp_(-CONFIG["CLAMP_NUM"], CONFIG["CLAMP_NUM"])

            real_images, labels = data_iter.__next__()
            current_batch_size = real_images.size(0)
            real_images = real_images.cuda() if CONFIG["GPU_NUMS"] > 0 else real_images
            x.data.resize_as_(real_images).copy_(real_images)
            errD_real = NetD(x)
            errD_real.backward(one)

            z.data.resize_(current_batch_size, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
            z_volatile = t.autograd.Variable(z.data, volatile = True)
            x_fake = t.autograd.Variable(NetG(z_volatile).data)
            errD_fake = NetD(x_fake)
            errD_fake.backward(one_neg)

            errD = (errD_real - errD_fake)
            optimizerD.step()

            t = t + 1
            i = i + 1

        for p in NetD.parameters():
            p.requires_grad = False

        NetG.zero_grad()

        # Sample fake data
        z.data.resize_(CONFIG["BATCH_SIZE"], CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
        x_fake = NetG(z)
        # Generator Loss
        errG = NetD(x_fake)
        errG.backward(one)
        optimizerG.step()

        bar.show(epoch, errD.item(), errG.item())

    fake_test = NetG(z_test)
    tv.utils.save_image(fake_test.data, 'outputs/Cat_%03d.png' % epoch, nrow=10, normalize=True)
