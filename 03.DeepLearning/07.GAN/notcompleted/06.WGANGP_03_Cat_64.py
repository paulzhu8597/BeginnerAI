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
    "LEARNING_RATE" : .0001
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
x_both = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["IMAGE_CHANNEL"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
z = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["NOISE_DIM"], 1, 1)
u = t.FloatTensor(CONFIG["BATCH_SIZE"], 1, 1, 1)
z_test = t.FloatTensor(100, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
grad_outputs = t.ones(CONFIG["BATCH_SIZE"])
one = t.FloatTensor([1])
one_neg = one * -1

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    x = x.cuda()
    z = z.cuda()
    u = u.cuda()
    z_test = z_test.cuda()
    grad_outputs = grad_outputs.cuda()
    one, one_neg = one.cuda(), one_neg.cuda()

x = t.autograd.Variable(x)
z = t.autograd.Variable(z)
z_test = t.autograd.Variable(z_test)

optimizerD = t.optim.Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0, .9))
optimizerG = t.optim.Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0, .9))

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(dataset), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    for i, (images, labels) in enumerate(dataset, 0):
        for p in NetD.parameters():
            p.requires_grad = True

        NetD.zero_grad()

        real_images = images.cuda() if CONFIG["GPU_NUMS"] > 0 else images
        x.data.copy_(real_images)
        errD_real = NetD(x)
        errD_real = errD_real.mean()
        errD_real.backward(one_neg)

        z.data.normal_(0, 1)
        z_volatile = t.autograd.Variable(z.data, volatile = True)
        x_fake = t.autograd.Variable(NetG(z_volatile).data)
        errD_fake = NetD(x_fake)
        errD_fake = errD_fake.mean()
        errD_fake.backward(one)

        u.uniform_(0, 1)
        x_both = x.data*u + x_fake.data*(1-u)
        x_both = x_both.cuda() if CONFIG["GPU_NUMS"] > 0 else x_both
        x_both = t.autograd.Variable(x_both, requires_grad=True)
        grad = t.autograd.grad(outputs=NetD(x_both), inputs=x_both, grad_outputs=grad_outputs, retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_penalty = 10 * ((grad.norm(2, 1).norm(2,1).norm(2,1) - 1) ** 2).mean()
        grad_penalty.backward()
        errD_penalty = errD_fake - errD_real + grad_penalty
        errD = errD_fake - errD_real
        optimizerD.step()
        errG = errD

        if (i + 1) % 5 == 0:
            for p in NetD.parameters():
                p.requires_grad = False

            NetG.zero_grad()

            z.data.normal_(0, 1)
            x_fake = NetG(z)
            errG = NetD(x_fake)
            errG = errG.mean()
            errG.backward(one_neg)
            optimizerG.step()
        bar.show(epoch, errD.item(), errG.item())
    fake_test = NetG(z_test)
    tv.utils.save_image(fake_test.data, 'outputs/Cat_%03d.png' % epoch, nrow=10, normalize=True)
