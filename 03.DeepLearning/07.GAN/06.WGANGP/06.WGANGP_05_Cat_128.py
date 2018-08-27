import torch as t
import torchvision as tv
import numpy
import torch.utils.data as t_data
import torch.autograd as t_auto
import torch.optim as t_optim

import lib.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/Cat128/",
    "EPOCH" : 100,
    "GPU_NUMS" : 1,
    "BATCH_SIZE" : 64,
    "NOISE_DIM" : 100,
    "IMAGE_CHANNEL": 3,
    "IMAGE_SIZE" : 128,
    "LAMBDA" : 0.25,
    "LEARNING_RATE" : .0001
}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(CONFIG["NOISE_DIM"], 1024, 4, 1, 0, bias=False),
            t.nn.SELU(True)
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            t.nn.SELU(True)
        )

        self.deconv3 = t.nn.Sequential(
            t.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            t.nn.SELU(True)
        )

        self.deconv4 = t.nn.Sequential(
            t.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            t.nn.SELU(True)
        )

        self.deconv5 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            t.nn.SELU(True)
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
            t.nn.SELU(True)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            t.nn.SELU(True)
        )

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            t.nn.SELU(True)
        )

        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            t.nn.SELU(True)
        )

        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            t.nn.SELU(True)
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

NetD = Discriminator()
NetG = Generator()
NetD.apply(weights_init)
NetG.apply(weights_init)

dataset = tv.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=tv.transforms.Compose([
    tv.transforms.Resize((CONFIG["IMAGE_SIZE"],CONFIG["IMAGE_SIZE"])),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = t_data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

x = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["IMAGE_CHANNEL"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
x_both = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["IMAGE_CHANNEL"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
z = t.FloatTensor(CONFIG["BATCH_SIZE"], CONFIG["NOISE_DIM"], 1, 1)
u = t.FloatTensor(CONFIG["BATCH_SIZE"], 1, 1, 1)
z_test = t.FloatTensor(100, CONFIG["NOISE_DIM"], 1, 1).normal_(0, 1)
grad_outputs = t.ones(CONFIG["BATCH_SIZE"])
one = t.FloatTensor([1])
one_neg = one * -1

# Everything cuda
if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    x = x.cuda()
    z = z.cuda()
    u = u.cuda()
    z_test = z_test.cuda()
    grad_outputs = grad_outputs.cuda()
    one, one_neg = one.cuda(), one_neg.cuda()

# Now Variables
x = t.autograd.Variable(x)
z = t.autograd.Variable(z)
z_test = t.autograd.Variable(z_test)

# Optimizer
optimizerD = t.optim.Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0, .9))
optimizerG = t.optim.Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0, .9))

def generate_random_sample():
    while True:
        random_indexes = numpy.random.choice(dataset.__len__(), size=CONFIG["BATCH_SIZE"], replace=False)
        batch = [dataset[i][0] for i in random_indexes]
        yield t.stack(batch, 0)
random_sample = generate_random_sample()

## Fitting model
bar = j_bar.ProgressBar(1, 5000, "D Loss%.3f;G Loss%.3f")
for i in range(1, 5000 + 1):
    for p in NetD.parameters():
        p.requires_grad = True

    for j in range(5):

        ########################
        # (1) Update D network #
        ########################

        NetD.zero_grad()

        # Sample real data
        real_images = random_sample.__next__()
        real_images = real_images.cuda() if CONFIG["GPU_NUMS"] > 0 else real_images
        x.data.copy_(real_images)
        # Discriminator Loss real
        errD_real = NetD(x)
        errD_real = errD_real.mean()
        errD_real.backward(one_neg)

        # Sample fake data
        z.data.normal_(0, 1)
        z_volatile = t.autograd.Variable(z.data, volatile = True)
        x_fake = t.autograd.Variable(NetG(z_volatile).data)
        # Discriminator Loss fake
        errD_fake = NetD(x_fake)
        errD_fake = errD_fake.mean()
        errD_fake.backward(one)

        # Gradient penalty
        u.uniform_(0, 1)
        x_both = x.data*u + x_fake.data*(1-u)
        x_both = x_both.cuda() if CONFIG["GPU_NUMS"] > 0 else x_both
        x_both = t.autograd.Variable(x_both, requires_grad=True)
        grad = t.autograd.grad(outputs=NetD(x_both), inputs=x_both, grad_outputs=grad_outputs, retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_penalty = 10*((grad.norm(2, 1).norm(2,1).norm(2,1) - 1) ** 2).mean()
        grad_penalty.backward()
        # Optimize
        errD_penalty = errD_fake - errD_real + grad_penalty
        errD = errD_fake - errD_real
        optimizerD.step()

    for p in NetD.parameters():
        p.requires_grad = False

    NetG.zero_grad()

    # Sample fake data
    z.data.normal_(0, 1)
    x_fake = NetG(z)
    # Generator Loss
    errG = NetD(x_fake)
    errG = errG.mean()
    #print(errG)
    errG.backward(one_neg)
    optimizerG.step()

    bar.show(i, errD.item(), errG.item())
    if (i%50) ==0:
        fake_test = NetG(z_test)
        tv.utils.save_image(fake_test.data, 'outputs/Cat_%03d.png' % (i/50), nrow=10,  normalize=True)
