import torch

from torch.nn import Module, Sequential, Linear, ReLU, Tanh, MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from lib.dataset.pytorch_dataset import MNISTDataSetForPytorch
from lib.ProgressBar import ProgressBar

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

GPU_NUMS = 0
EPOCH = 100
BATCH_SIZE = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = MNISTDataSetForPytorch(train=True, transform=img_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

class autoencoder(Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = Sequential(
            Linear(28 * 28, 128),
            ReLU(True),
            Linear(128, 64),
            ReLU(True), Linear(64, 12), ReLU(True), Linear(12, 3))
        self.decoder = Sequential(
            Linear(3, 12),
            ReLU(True),
            Linear(12, 64),
            ReLU(True),
            Linear(64, 128),
            ReLU(True), Linear(128, 28 * 28), Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder().cuda() if GPU_NUMS > 0 else autoencoder()
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

proBar = ProgressBar(EPOCH, len(train_loader), "Loss:%.3f")

for epoch in range(1, EPOCH):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda() if GPU_NUMS > 0 else Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proBar.show(epoch, loss.item())
    # ===================log========================

    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, 'output/image_{}.png'.format(epoch))

torch.save(model.state_dict(), 'sim_autoencoder.pth')