# coding=utf-8
import torch
import os

from torch.nn import Linear, DataParallel, Conv2d
from torchvision.transforms.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import SGD

from lib.models.yolov1 import YOLOv1Net, YOLOv1Loss, YOLOv1Detection
from lib.dataset.pytorch_dataset import yoloDataset
from lib.ProgressBar import ProgressBar

YOLOv1Config = {
    "GPU_NUMS" : 1,
    "EPOCHS" : 120,
    "IMAGE_SIZE" : 448,
    "IMAGE_CHANNEL" : 3,
    "ALPHA" : 0.1,
    "BATCH_SIZE" : 32,
    "DATA_PATH" : "/input/",
    "CELL_NUMS" : 7,
    "CLASS_NUMS" : 20,
    "BOXES_EACH_CELL" : 2,
    "LEARNING_RATE" : 0.001,
    "L_COORD" : 5,
    "L_NOOBJ" : 0.5,
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor']
}

FROM_TRAIN_ITER = 15

Net = YOLOv1Net(CFG=YOLOv1Config)

for m in Net.modules():
    if isinstance(m, Linear) or isinstance(m, Conv2d):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

if YOLOv1Config["GPU_NUMS"] > 0:
    Net = Net.cuda()

if YOLOv1Config["GPU_NUMS"] > 1 :
    Net = DataParallel(Net)

train_dataset = yoloDataset(root = os.path.join(YOLOv1Config["DATA_PATH"], "VOC2012", "JPEGImages"), list_file='utils/voc2012train.txt',
                            train=True, transform=[ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=YOLOv1Config["BATCH_SIZE"], shuffle=True)

criterion = YOLOv1Loss()
# 优化器
optimizer = SGD(Net.parameters(), lr=YOLOv1Config["LEARNING_RATE"], momentum=0.95, weight_decay=5e-4)
bar = ProgressBar(YOLOv1Config["EPOCHS"], len(train_loader), "Loss:%.3f", current_epoch=FROM_TRAIN_ITER)

if FROM_TRAIN_ITER > 1:
    Net.load_state_dict(torch.load("output/YoloV1_%d.pth" % (FROM_TRAIN_ITER - 1)))

for epoch in range(FROM_TRAIN_ITER, YOLOv1Config["EPOCHS"]):
    if epoch == 1:
        LEARNING_RATE = 0.0005
    if epoch == 2:
        LEARNING_RATE = 0.00075
    if epoch >= 3 and epoch < 80:
        LEARNING_RATE = 0.001
    if epoch >= 80 and epoch < 100:
        LEARNING_RATE = 0.0001
    if epoch >= 100:
        LEARNING_RATE = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE

    for i, (images, target) in enumerate(train_loader):
        images = Variable(images.cuda() if YOLOv1Config["GPU_NUMS"] > 0 else images)
        target = Variable(target.cuda() if YOLOv1Config["GPU_NUMS"] > 0 else target)

        optimizer.zero_grad()
        pred = Net(images)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        bar.show(epoch, loss.item())
    # 保存最新的模型
    torch.save(Net.state_dict(),"output/YoloV1_%d.pth" % epoch)

    predict = YOLOv1Detection(YOLOv1Config["CLASSES"], sourceImagePath="../testImages/demo.jpg", targetImagePath="outputs/", Net=Net)
    predict.imageShow()