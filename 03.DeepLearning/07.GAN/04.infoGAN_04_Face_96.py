import torch
import torchvision
import numpy as np

from lib.ProgressBar import ProgressBar

CONFIG={
    "DATA_PATH" : "/input/Faces/Eyeglasses",
    "NOISE_DIM" : 100,
    "Z_DIM" : 128,
    "CC_DIM" : 1,
    "DC_DIM" : 10,
    "IMAGE_SIZE" : 96,
    "IMAGE_CHANNEL" : 3,
    "BATCH_SIZE" : 64,
    "GPU_NUMS" : 1,
    "EPOCH" : 100,
    "CONTINUOUS_WEIGHT" : 0.5
}

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(CONFIG["Z_DIM"] + CONFIG["CC_DIM"] + CONFIG["DC_DIM"], 1024, 4, 1, 0)
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, CONFIG["IMAGE_CHANNEL"], 4, 3, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        output = x.view(x.size(0), x.size(1), 1, 1)
        output = self.deconv1(output)
        output = self.deconv2(output)
        output = self.deconv3(output)
        output = self.deconv4(output)
        output = self.deconv5(output)

        return output

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(CONFIG["IMAGE_CHANNEL"], 128, 4, 3, 1),
            torch.nn.LeakyReLU(0.1, True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 4, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1, True)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 4, 2, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, True)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 4, 2, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1, True)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1 + CONFIG["CC_DIM"] + CONFIG["DC_DIM"], 4, 1, 0)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.squeeze()

        output[:, 0] = torch.nn.functional.sigmoid(output[:, 0].clone())

        output[:, CONFIG["CC_DIM"] + 1 : CONFIG["CC_DIM"] + 1 + CONFIG["DC_DIM"]] = torch.nn.functional.softmax(output[:, CONFIG["CC_DIM"] + 1 : CONFIG["CC_DIM"] + 1 + CONFIG["DC_DIM"]].clone())

        return output

transform = torchvision.transforms.Compose([
    torchvision.transforms.Scale((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

NetG = Generator()
NetD = Discriminator()
if CONFIG["GPU_NUMS"] > 0:
    NetD = NetD.cuda()
    NetG = NetG.cuda()

g_optimizer = torch.optim.Adam(NetG.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(NetD.parameters(), lr=0.001, betas=(0.5, 0.999))

fixed_noise = torch.Tensor(np.zeros((CONFIG["NOISE_DIM"], CONFIG["Z_DIM"])))
fixed_noise_var = torch.autograd.Variable(fixed_noise.cuda() if CONFIG["GPU_NUMS"] > 0 else fixed_noise)
tmp = np.zeros((CONFIG["NOISE_DIM"], CONFIG["CC_DIM"]))
for k in range(10):
    tmp[k * 10:(k + 1) * 10, 0] = np.linspace(-2, 2, 10)
fixed_cc = torch.Tensor(tmp)
fixed_cc_var =torch.autograd.Variable(fixed_cc.cuda() if CONFIG["GPU_NUMS"] > 0 else fixed_cc)

tmp = np.zeros((CONFIG["NOISE_DIM"], CONFIG["DC_DIM"]))
for k in range(10):
    tmp[k * 10 : (k + 1) * 10, k] = 1
fixed_dc = torch.Tensor(tmp)
fixed_dc_var = torch.autograd.Variable(fixed_dc.cuda() if CONFIG["GPU_NUMS"] > 0 else fixed_dc)
bar = ProgressBar(CONFIG["EPOCH"], len(data_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    for i, images in enumerate(data_loader):
        images = torch.autograd.Variable(images.cuda() if CONFIG["GPU_NUMS"] > 0 else images)

        mini_batch = images.size(0)

        cc = torch.Tensor(np.random.randn(mini_batch, CONFIG["CC_DIM"]) * 0.5 + 0.0)
        cc_var = torch.autograd.Variable(cc.cuda() if CONFIG["GPU_NUMS"] > 0 else cc)

        codes=[]
        code = np.zeros((mini_batch, CONFIG["DC_DIM"]))
        random_cate = np.random.randint(0, CONFIG["DC_DIM"], mini_batch)
        code[range(mini_batch), random_cate] = 1
        codes.append(code)
        codes = np.concatenate(codes,1)
        dc = torch.Tensor(codes)
        dc_var = torch.autograd.Variable(dc.cuda() if CONFIG["GPU_NUMS"] > 0 else dc)

        noise = torch.randn(mini_batch, CONFIG["Z_DIM"])
        noise_var = torch.autograd.Variable(noise.cuda() if CONFIG["GPU_NUMS"] > 0 else noise)
        fake_images = NetG(torch.cat((noise_var, cc_var, dc_var),1))
        d_output_real = NetD(images)
        d_output_fake = NetD(fake_images)

        d_loss_a = -torch.mean(torch.log(d_output_real[:,0]) + torch.log(1 - d_output_fake[:,0]))

        # Mutual Information Loss
        output_cc = d_output_fake[:, 1:1+CONFIG["CC_DIM"]]
        output_dc = d_output_fake[:, 1+CONFIG["CC_DIM"]:]
        d_loss_cc = torch.mean((((output_cc - 0.0) / 0.5) ** 2))
        d_loss_dc = -(torch.mean(torch.sum(dc * output_dc, 1)) + torch.mean(torch.sum(dc * dc, 1)))

        d_loss = d_loss_a + 0.5 * d_loss_cc + 1.0 * d_loss_dc

        # Optimization
        NetD.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        # ===================== Train G =====================#
        # Fake -> Real
        g_loss_a = -torch.mean(torch.log(d_output_fake[:,0]))

        g_loss = g_loss_a + CONFIG["CONTINUOUS_WEIGHT"] * d_loss_cc + 1.0 * d_loss_dc

        # Optimization
        NetG.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        bar.show(epoch, d_loss.item(), g_loss.item())

    fake_images = NetG(torch.cat((fixed_noise_var, fixed_cc_var, fixed_dc_var), 1))
    torchvision.utils.save_image(fake_images.data, "outputs/cifar10_%03d.png" % epoch, nrow=10)