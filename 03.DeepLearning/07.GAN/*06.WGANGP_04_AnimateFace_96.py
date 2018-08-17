import torch as t
import torchvision as tv

import torch.utils.data as t_data
import torch.autograd as t_auto
import torch.optim as t_optim

import lib.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/Faces/SquareImages/",
    "EPOCH" : 100,
    "GPU_NUMS" : 1,
    "BATCH_SIZE" : 64,
    "NOISE_DIM" : 62,
    "IMAGE_CHANNEL": 3,
    "IMAGE_SIZE" : 96,
    "LAMBDA" : 0.25,
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
            t.nn.Linear(CONFIG["NOISE_DIM"], 1024),
            t.nn.BatchNorm1d(1024),
            t.nn.ReLU()
        )

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(1024, 128 * 24 * 24),
            t.nn.BatchNorm1d(128 * 24 * 24),
            t.nn.ReLU()
        )

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU()
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(64, CONFIG["IMAGE_CHANNEL"], 4, 2, 1),
            t.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = output.view(-1, 128, 24, 24)
        output = self.deconv1(output)
        output = self.deconv2(output)

        return output

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(CONFIG["IMAGE_CHANNEL"], 64, 4, 2, 1),
            t.nn.LeakyReLU(0.2)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 4, 2, 1),
            t.nn.BatchNorm2d(128),
            t.nn.LeakyReLU(0.2)
        )

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(128 * 24 * 24, 1024),
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
        output = output.view(-1, 128 * 24 * 24)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

NetD = Discriminator()
NetG = Generator()

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()

optimizerD = t_optim.Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))
optimizerG = t_optim.Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))

dataset = tv.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=tv.transforms.Compose([
    # tv.transforms.Resize(CONFIG["IMAGE_SIZE"]),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = t_data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

fix_noise = t.randn(100, CONFIG["NOISE_DIM"])
fix_noise_var = t_auto.Variable(fix_noise.cuda() if CONFIG["GPU_NUMS"] > 0 else fix_noise)

def calc_gradient_penalty(netD, real_data, fake_data, mini_batch):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = t.rand(mini_batch, 1)
    alpha = alpha.expand(mini_batch, int(real_data.nelement()/mini_batch)).contiguous().view(mini_batch, CONFIG["IMAGE_CHANNEL"],
                                                                                             CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
    alpha = alpha.cuda() if CONFIG["GPU_NUMS"] > 0 else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if CONFIG["GPU_NUMS"] > 0:
        interpolates = interpolates.cuda()
    interpolates = t_auto.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = t_auto.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=t.ones(disc_interpolates.size()).cuda() if CONFIG["GPU_NUMS"] > 0 else t.ones(
                                disc_interpolates.size()),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * CONFIG["LAMBDA"]
    return gradient_penalty

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]
        noise = t.rand(mini_batch, CONFIG["NOISE_DIM"])
        real_var = t_auto.Variable(image.cuda() if CONFIG["GPU_NUMS"] > 0 else image)
        noise_var = t_auto.Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)
        alpha = t.rand(mini_batch).cuda() if CONFIG["GPU_NUMS"] > 0 else t.rand(mini_batch)

        optimizerD.zero_grad()

        D_real = NetD(real_var)
        D_real_loss = -t.mean(D_real)


        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = t.mean(D_fake)

        gradient_penalty = calc_gradient_penalty(NetD, real_var.data, G_.data, mini_batch)
        D_loss = D_real_loss + D_fake_loss + gradient_penalty
        D_loss.backward()
        optimizerD.step()

        G_loss = D_loss
        if ((index+1) % 5) == 0:
            # update G network
            optimizerG.zero_grad()

            G_ = NetG(noise_var)
            D_fake = NetD(G_)
            G_loss = -t.mean(D_fake)

            G_loss.backward()
            optimizerG.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/Face_%03d.png' % epoch,nrow=10)

