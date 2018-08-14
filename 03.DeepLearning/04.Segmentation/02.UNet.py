import torch

from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lib.models.seg import UNet, cross_entropy2d
from lib.dataset.pytorch_dataset import VOCSegDataSet
from lib.utils.utils_fcn import Compose, RandomHorizontallyFlip, RandomRotate
from lib.ProgressBar import ProgressBar

UNetConfig = {
    "CLASS_NUMS" : 21,
    "CLASSES" : ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "IMAGE_SIZE" : 416,
    "EPOCHS" : 120,
    "IMAGE_CHANNEL" : 3,
    "GPU_NUMS" : 0,
    "LEARNING_RATE" : 1.0e-10,
    "BATCH_SIZE" : 1,
    "MOMENTUM" : 0.99,
    "WEIGHT_DECAY" : 0.0005,
    "DATA_PATH" : "/input/",
    "FEATURE_SCALE" : 4
}

'''
Model
'''
model = UNet(cfg=UNetConfig)

if UNet["GPU_NUMS"] > 0:
    model = model.cuda()

optimizer = SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=5e-4)
'''
Loss
'''
loss_fn = cross_entropy2d

'''
Data
'''

data_loader = VOCSegDataSet(is_transform=True, img_size=(UNet["IMAGE_SIZE"], UNet["IMAGE_SIZE"]),
                            augmentations=Compose([RandomRotate(10),
                                                   RandomHorizontallyFlip()]), img_norm=True)

train_loader = DataLoader(data_loader, batch_size=UNet["BATCH_SIZE"], shuffle=True)

'''
Train
'''
bar = ProgressBar(UNet["EPOCHS"], len(train_loader), "Loss:%.3f")
for epoch in range(1, UNet["EPOCHS"]):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda() if UNet["GPU_NUMS"] > 0 else images)
        labels = Variable(labels.cuda() if UNet["GPU_NUMS"] > 0 else labels)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels)

        loss.backward()
        optimizer.step()

        bar.show(epoch, loss.item())
    torch.save(model.state_dict(), "UNet_%03d.pth" % epoch)



