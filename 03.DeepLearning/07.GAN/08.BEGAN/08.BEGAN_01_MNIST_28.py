import torch as t
import torchvision as tv

import torch.utils.data as t_data
import torch.autograd as t_auto
import torch.optim as t_optim

import lib.dataset.pytorch_dataset as j_data
import lib.ProgressBar as j_bar

CONFIG = {
    "EPOCH" : 100,
    "GPU_NUMS" : 0,
    "BATCH_SIZE" : 64,
    "NOISE_DIM" : 62,
    "IMAGE_CHANNEL": 1,
    "IMAGE_SIZE" : 28,
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

        self.model1 = t.nn.Sequential(
            t.nn.Linear(in_features=CONFIG["NOISE_DIM"],out_features=1024),
            t.nn.BatchNorm1d(num_features=1024),
            t.nn.ReLU(),
            t.nn.Linear(in_features=1024, out_features=128 * 7 * 7),
            t.nn.BatchNorm1d(num_features=128*7*7),
            t.nn.ReLU()
        )

        self.model2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            t.nn.BatchNorm2d(num_features=64),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(in_channels=64, out_channels=CONFIG["IMAGE_CHANNEL"], kernel_size=4, stride=2, padding=1),
            t.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        network = self.model1(x)

        network = network.view(-1, 128, 7, 7)
        network = self.model2(network)

        return network

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels=CONFIG["IMAGE_CHANNEL"], out_channels=64, kernel_size=4, stride=2, padding=1),
            t.nn.ReLU()
        )
        self.model2 = t.nn.Sequential(
            t.nn.Linear(in_features=64 * 14 * 14, out_features=32),
            t.nn.BatchNorm1d(num_features=32),
            t.nn.ReLU(),
            t.nn.Linear(in_features=32, out_features= 64 * 14 * 14),
            t.nn.BatchNorm1d(num_features=64 * 14 * 14),
            t.nn.ReLU()
        )
        self.model3 = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        )

        initialize_weights(self)

    def forward(self, x):
        network = self.model1(x)

        network = network.view(network.size()[0], -1)

        network = self.model2(network)

        network = network.view(-1, 64, 14,14)

        network = self.model3(network)

        return network

NetD = Discriminator()
NetG = Generator()
if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()

optimizerD = t_optim.Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))
optimizerG = t_optim.Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))

dataset = j_data.MNISTDataSetForPytorch(transform=tv.transforms.Compose([
    # tv.transforms.Resize(CONFIG["IMAGE_SIZE"]),
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = t_data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

fix_noise = t.randn(100, CONFIG["NOISE_DIM"])
fix_noise_var = t_auto.Variable(fix_noise.cuda() if CONFIG["GPU_NUMS"] > 0 else fix_noise)

k = 0.
gamma = 0.75
lambda_ = 0.001

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    for index, (image, label) in enumerate(train_loader):
        mini_batch = image.shape[0]

        noise = t.rand(mini_batch, CONFIG["NOISE_DIM"])

        real_var = t_auto.Variable(image.cuda() if CONFIG["GPU_NUMS"] > 0 else image)
        noise_var = t_auto.Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)

        label_real_var = t_auto.Variable(t.ones(mini_batch, 1).cuda()  if CONFIG["GPU_NUMS"] > 0 else t.ones(mini_batch, 1))
        label_fake_var = t_auto.Variable(t.zeros(mini_batch, 1).cuda() if CONFIG["GPU_NUMS"] > 0 else t.zeros(mini_batch, 1))

        NetD.zero_grad()

        D_real = NetD(real_var)
        D_real_loss = t.mean(t.abs(D_real - real_var))

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = t.mean(t.abs(D_fake - G_))

        D_loss = D_real_loss - k * D_fake_loss
        D_loss.backward()
        optimizerD.step()

        NetG.zero_grad()

        G_ = NetG(noise_var)
        D_fake = NetD(G_)
        D_fake_loss = t.mean(t.abs(D_fake - G_))
        G_loss = D_fake_loss

        G_loss.backward()
        optimizerG.step()

        temp_M = D_real_loss + t.abs(gamma * D_real_loss - D_fake_loss)

        temp_K = k + lambda_ * (gamma * D_real_loss - D_fake_loss)
        temp_K = temp_K.data[0]
        k = min(max(temp_K, 0), 1)
        M = temp_M.data[0]

        bar.show(epoch, D_loss.item(), G_loss.item())

    fake_u=NetG(fix_noise_var)
    tv.utils.save_image(fake_u.data,'outputs/mnist_%03d.png' % epoch,nrow=10)

