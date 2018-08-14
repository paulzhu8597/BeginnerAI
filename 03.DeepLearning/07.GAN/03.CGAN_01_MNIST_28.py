import torch
import torchvision

from torch.nn import Module, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, LeakyReLU, Tanh, Sigmoid, BCELoss,\
    Sequential, Linear, BatchNorm1d
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.transforms import Compose,Scale,ToTensor, Normalize

from lib.ProgressBar import ProgressBar
from lib.dataset.pytorch_dataset import MNISTDataSetForPytorch

CONFIG = {
    "DATA_PATH" : "/input/mnist.npz",
    "EPOCHS" : 100,
    "BATCH_SIZE" : 128,
    "GPU_NUMS" : 1,
    "IMAGE_SIZE" : 28,
    "IMAGE_CHANNEL" : 1,
    "NOISE_DIM" : 100,
    "LEARNING_RATE" : 2e-4,
    "BETA1" : 0.5
}

def normal_init(net):
    for m in net.modules():
        if isinstance(m, Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = Sequential(
            Linear(CONFIG["NOISE_DIM"] + 10, 1024),
            BatchNorm1d(1024),
            ReLU(),
            Linear(1024, 128 * 7 * 7),
            BatchNorm1d(128 * 7 * 7),
            ReLU(),
        )
        self.deconv = Sequential(
            ConvTranspose2d(128, 64, 4, 2, 1),
            BatchNorm2d(64),
            ReLU(),
            ConvTranspose2d(64, CONFIG["IMAGE_CHANNEL"], 4, 2, 1),
            Tanh(),
        )

        normal_init(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = x.view(-1,CONFIG["NOISE_DIM"] + 10)
        x = self.fc(x)
        x = x.view(-1, 128, (CONFIG["IMAGE_SIZE"] // 4), (CONFIG["IMAGE_SIZE"] // 4))
        x = self.deconv(x)

        return x

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = Sequential(
            Conv2d(CONFIG["IMAGE_CHANNEL"] + 10, 64, 4, 2, 1),
            LeakyReLU(0.2),
            Conv2d(64, 128, 4, 2, 1),
            BatchNorm2d(128),
            LeakyReLU(0.2),
        )
        self.fc = Sequential(
            Linear(128 * 7 * 7, 1024),
            BatchNorm1d(1024),
            LeakyReLU(0.2),
            Linear(1024, 1),
            Sigmoid(),
        )
        normal_init(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)

        return x

NetG = Generator()
NetD = Discriminator()
BCE_LOSS = BCELoss()
G_optimizer = Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))
D_optimizer = Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))

if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    BCE_LOSS = BCE_LOSS.cuda()

transform = Compose([
    ToTensor()
])
train_loader = torch.utils.data.DataLoader(
    MNISTDataSetForPytorch(train=True, transform=transform),
    batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

fill = torch.zeros([10, 10, CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"]])
for i in range(10):
    fill[i, i, :, :] = 1

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)
fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)

with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda() if CONFIG["GPU_NUMS"] > 0 else fixed_z_, volatile=True)
    fixed_y_label_ = Variable(fixed_y_label_.cuda() if CONFIG["GPU_NUMS"] > 0 else fixed_y_label_, volatile=True)

bar = ProgressBar(CONFIG["EPOCHS"], len(train_loader), "D Loss:%.3f, G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    if epoch % 30 == 0:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10

    for img_real, label_real in train_loader:
        mini_batch = label_real.shape[0]

        label_true_var  = Variable(torch.ones(mini_batch).cuda() if CONFIG["GPU_NUMS"] > 0 else torch.ones(mini_batch))
        label_false_var = Variable(torch.zeros(mini_batch).cuda() if CONFIG["GPU_NUMS"] > 0 else torch.zeros(mini_batch))

        NetD.zero_grad()
        label_real = label_real.squeeze().type(torch.LongTensor)
        label_real = fill[label_real]

        image_var = Variable(img_real.cuda() if CONFIG["GPU_NUMS"] > 0 else img_real)
        label_var = Variable(label_real.cuda() if CONFIG["GPU_NUMS"] > 0 else label_real)

        d_result = NetD(image_var, label_var)
        d_result = d_result.squeeze()
        D_LOSS_REAL = BCE_LOSS(d_result, label_true_var)

        img_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        img_fake_var = Variable(img_fake.cuda() if CONFIG["GPU_NUMS"] > 0 else img_fake)
        label_fake_G_var = Variable(onehot[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else onehot[label_fake])
        label_fake_D_var = Variable(fill[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else fill[label_fake])
        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        D_LOSS_FAKE = BCELoss()(d_result, label_false_var)

        D_train_loss = D_LOSS_REAL + D_LOSS_FAKE
        D_train_loss.backward()
        D_optimizer.step()

        NetG.zero_grad()
        img_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        img_fake_var = Variable(img_fake.cuda() if CONFIG["GPU_NUMS"] > 0 else img_fake)
        label_fake_G_var = Variable(onehot[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else onehot[label_fake])
        label_fake_D_var = Variable(fill[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else fill[label_fake])
        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        G_train_loss= BCELoss()(d_result, label_true_var)
        G_train_loss.backward()
        G_optimizer.step()

        bar.show(epoch, D_train_loss.item(), G_train_loss.item())

    test_images = NetG(fixed_z_, fixed_y_label_)

    torchvision.utils.save_image(test_images.data[:100],'outputs/mnist_%03d.png' % (epoch),nrow=10,
                                 normalize=True,range=(-1,1), padding=0)