import torch

from skimage.io import imread
from torch.nn import Module, Linear, BCELoss
from torch.autograd import Variable
from torch.nn.functional import leaky_relu, sigmoid
from torch.optim import Adadelta


from utils.sampler import generate_lut,sample_2d
from utils.visualize import GANDemoVisualizer
from lib.utils.progressbar.ProgressBar import ProgressBar

PHRASE = "TRAIN"
DIMENSION = 2
iterations = 3000
bs = 2000
GPU_NUMS = 1
z_dim = 2
input_path = "inputs/Z.jpg"

density_img = imread(input_path, True)
lut_2d = generate_lut(density_img)

visualizer = GANDemoVisualizer('GAN 2D Example Visualization of {}'.format(input_path))

class SimpleMLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.map1 = Linear(input_size, hidden_size)
        self.map2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = leaky_relu(self.map1(x), 0.1)
        return sigmoid(self.map2(x))

class DeepMLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLP, self).__init__()
        self.map1 = Linear(input_size, hidden_size)
        self.map2 = Linear(hidden_size, hidden_size)
        self.map3 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = leaky_relu(self.map1(x), 0.1)
        x = leaky_relu(self.map2(x), 0.1)
        return sigmoid(self.map3(x))
generator = SimpleMLP(input_size=z_dim, hidden_size=50, output_size=DIMENSION)
discriminator = SimpleMLP(input_size=DIMENSION, hidden_size=100, output_size=1)
if GPU_NUMS > 0:
    generator.cuda()
    discriminator.cuda()
criterion = BCELoss()

d_optimizer = Adadelta(discriminator.parameters(), lr=1)
g_optimizer = Adadelta(generator.parameters(), lr=1)
progBar = ProgressBar(1, iterations, "D Loss:(real/fake) %.3f/%.3f,G Loss:%.3f")
for train_iter in range(1, iterations + 1):
    for d_index in range(3):
        # 1. Train D on real+fake
        discriminator.zero_grad()

        #  1A: Train D on real
        real_samples = sample_2d(lut_2d, bs)
        d_real_data = Variable(torch.Tensor(real_samples))
        if GPU_NUMS > 0:
            d_real_data = d_real_data.cuda()
        d_real_decision = discriminator(d_real_data)
        labels = Variable(torch.ones(bs))
        if GPU_NUMS > 0:
            labels = labels.cuda()
        d_real_loss = criterion(d_real_decision, labels)  # ones = true

        #  1B: Train D on fake
        latent_samples = torch.randn(bs, z_dim)
        d_gen_input = Variable(latent_samples)
        if GPU_NUMS > 0:
            d_gen_input = d_gen_input.cuda()
        d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = discriminator(d_fake_data)
        labels = Variable(torch.zeros(bs))
        if GPU_NUMS > 0:
            labels = labels.cuda()
        d_fake_loss = criterion(d_fake_decision, labels)  # zeros = fake

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(1):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        generator.zero_grad()

        latent_samples = torch.randn(bs, z_dim)
        g_gen_input = Variable(latent_samples)
        if GPU_NUMS > 0:
            g_gen_input = g_gen_input.cuda()
        g_fake_data = generator(g_gen_input)
        g_fake_decision = discriminator(g_fake_data)
        labels = Variable(torch.ones(bs))
        if GPU_NUMS > 0:
            labels = labels.cuda()
        g_loss = criterion(g_fake_decision, labels)  # we want to fool, so pretend it's all genuine

        g_loss.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    loss_d_real = d_real_loss.item()
    loss_d_fake = d_fake_loss.item()
    loss_g = g_loss.item()

    progBar.show(loss_d_real, loss_d_fake, loss_g)
    if train_iter == 1 or train_iter % 100 == 0:
        msg = 'Iteration {}: D_loss(real/fake): {:.6g}/{:.6g} G_loss: {:.6g}'.format(train_iter, loss_d_real, loss_d_fake, loss_g)

        gen_samples = g_fake_data.data.cpu().numpy() if GPU_NUMS > 0 else g_fake_data.data.numpy()

        visualizer.draw(real_samples, gen_samples, msg, show=False)
        visualizer.savefig('output/Pytorch_Z_%04d.png' % train_iter)

torch.save(generator.state_dict(), "output/GAN_Z_Pytorch_Generator.pth")
