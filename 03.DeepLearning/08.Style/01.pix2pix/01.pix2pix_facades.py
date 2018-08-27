import torch as t
import torchvision as tv
import torch.autograd as t_auto
import torch.optim as t_optim
import os
import lib.dataset.pytorch_dataset as j_data
import lib.ProgressBar as j_bar
import lib.utils.utils_style as j_utils
CONFIG = {
    "GPU_NUM" : 1,
    "EPOCH" : 200,
    "IMAGE_CHANNEL" : 3,
    "IMAGE_SIZE" : 256,
    "BATCH_SIZE" : 1
}

class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_filters = 64

        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(CONFIG["IMAGE_CHANNEL"], self.num_filters, 4, 2, 1)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters, self.num_filters * 2, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 2)
        )
        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 2, self.num_filters * 4, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 4)
        )
        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 4, self.num_filters * 8, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 8)
        )
        self.conv567 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 8, self.num_filters * 8, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 8)
        )
        self.conv8 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 4, self.num_filters * 8, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
        )

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 8, self.num_filters * 8, 4, 2, 1),
            t.nn.ReLU(True),
            t.nn.BatchNorm2d(self.num_filters * 8),
            t.nn.Dropout(0.5)
        )
        self.deconv23 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 8 * 2, self.num_filters * 8, 4, 2, 1),
            t.nn.ReLU(True),
            t.nn.BatchNorm2d(self.num_filters * 8),
            t.nn.Dropout(0.5)
        )
        self.deconv4 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 8 * 2, self.num_filters * 8, 4, 2, 1),
            t.nn.ReLU(True),
            t.nn.BatchNorm2d(self.num_filters * 8),
        )
        self.deconv5 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 8 * 2, self.num_filters * 4, 4, 2, 1),
            t.nn.ReLU(True),
            t.nn.BatchNorm2d(self.num_filters * 4),
        )
        self.deconv6 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 4 * 2, self.num_filters * 2, 4, 2, 1),
            t.nn.ReLU(True),
            t.nn.BatchNorm2d(self.num_filters * 2),
        )
        self.deconv7 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 2 * 2, self.num_filters, 4, 2, 1),
            t.nn.ReLU(True),
            t.nn.BatchNorm2d(self.num_filters),
        )
        self.deconv8 = t.nn.Sequential(
            t.nn.ConvTranspose2d(self.num_filters * 2, 3, 4, 2, 1),
            t.nn.ReLU(True),
        )

    def forward(self, input):
        enc1 = self.conv1(input)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv567(enc4)
        enc6 = self.conv567(enc5)
        enc7 = self.conv567(enc6)
        enc8 = self.conv8(enc7)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        dec1 = t.cat([dec1, enc7], 1)
        dec2 = self.deconv23(dec1)
        dec2 = t.cat([dec2, enc6], 1)
        dec3 = self.deconv23(dec2)
        dec3 = t.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        dec4 = t.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        dec5 = t.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        dec6 = t.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6)
        dec7 = t.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7)
        out = t.nn.Tanh()(dec8)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.BatchNorm2d) or isinstance(m, t.nn.ConvTranspose2d):
                t.nn.init.normal(m.conv.weight, mean, std)

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_filters = 64
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(6, self.num_filters , 4, 2, 1),
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters, self.num_filters * 2, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 2)
        )
        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 2, self.num_filters * 4, 4, 2, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 4)
        )
        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 4, self.num_filters * 8, 4, 1, 1),
            t.nn.LeakyReLU(0.2, True),
            t.nn.BatchNorm2d(self.num_filters * 4)
        )
        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(self.num_filters * 8, 1, 4, 1, 1),
            t.nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, label):
        x = t.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = t.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.BatchNorm2d):
                t.nn.init.normal(m.conv.weight, mean, std)
'''
构造网络
'''
Net_G = Generator()
Net_D = Discriminator()
Net_G.normal_weight_init(mean=0.0, std=0.02)
Net_D.normal_weight_init(mean=0.0, std=0.02)

BCE_loss = t.nn.BCELoss()
L1_loss = t.nn.L1Loss()

if CONFIG["GPU_NUM"] > 0:
    Net_D.cuda()
    Net_G.cuda()
    BCE_loss = BCE_loss.cuda()
    L1_loss = L1_loss.cuda()
G_optimizer = t_optim.Adam(Net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = t_optim.Adam(Net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

'''
读入数据
'''
if not os.path.exists("output"):
    os.mkdir("output")

train_set = j_data.DataSetFromFolderForPix2Pix(os.path.join("/input/facades_fixed", "train"))
test_set  = j_data.DataSetFromFolderForPix2Pix(os.path.join("/input/facades_fixed", "test"))
train_data_loader = t.utils.data.DataLoader(dataset=train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
test_data_loader  = t.utils.data.DataLoader(dataset=test_set,  batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

test_input, test_target = test_data_loader.__iter__().__next__()

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_data_loader), "D loss:%.3f;G loss:%.3f")
for epoch in range(1, CONFIG["EPOCH"] + 1):
    for i, (input, target) in enumerate(train_data_loader):
        x_ = t_auto.Variable(input.cuda() if CONFIG["GPU_NUM"] > 0 else input)
        y_ = t_auto.Variable(target.cuda() if CONFIG["GPU_NUM"] > 0 else target)

        # Train discriminator with real data
        D_real_decision = Net_D(x_, y_).squeeze()
        real_ = t_auto.Variable(t.ones(D_real_decision.size()).cuda() if CONFIG["GPU_NUM"] > 0 else t.ones(D_real_decision.size()))
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = Net_G(x_)
        D_fake_decision = Net_D(x_, gen_image).squeeze()
        fake_ = t_auto.Variable(t.zeros(D_fake_decision.size()).cuda() if CONFIG["GPU_NUM"] > 0 else t.zeros(D_fake_decision.size()))
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        Net_D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = Net_G(x_)
        D_fake_decision = Net_D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = 100 * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        Net_G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        bar.show(epoch, D_loss.item(), G_loss.item())

    gen_image = Net_G(t_auto.Variable(test_input.cuda() if CONFIG["GPU_NUM"] > 0 else test_input))
    gen_image = gen_image.cpu().data
    j_utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir='outputs/')
