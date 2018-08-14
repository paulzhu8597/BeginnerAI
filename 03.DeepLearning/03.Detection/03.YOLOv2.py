import torch

from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from lib.dataset.pytorch_dataset import SSDDataSetForPytorch
from lib.models.yolov2 import YOLOv2Net, YoloLoss, YOLOv2Detection
from lib.utils.models import Augmentation, AnnotationTransform
from lib.utils.functions import weights_init, detection_collate
from lib.ProgressBar import ProgressBar

YOLOv2Config = {
    "ANCHORS" : [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
    "CLASS_NUMS" : 20,
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "FEAT_SIZE" : 13,
    "EVAL_NMS_THRESHOLD" : 0.3,
    "EVAL_SCORE_THRESHOLD" : 1e-4,
    "SCORE_THRESHOLD" : 0.5,
    "NMS_THRESHOLD" : 0.3,
    "IOU_THRESHOLD" : 0.6,
    "GPU_NUMS" : 0,
    "LEARNING_RATE" : 1e-4,
    "BATCH_SIZE" : 32,
    "MOMENTUM" : 0.9,
    "WEIGHT_DECAY" : 1e-4,
    "DATA_PATH" : "/input/",
    "IMAGE_SIZE" : 416,
    "EPOCHS" : 120,
    "IMAGE_CHANNEL" : 3,
    "NO_OBJECT_SCALE" : 1,
    "OBJECT_SCALE" : 5,
    "CLASS_SCALE" : 1,
    "COORD_SCALE" : 1
}

warm_lr = YOLOv2Config["LEARNING_RATE"] / YOLOv2Config["BATCH_SIZE"]

net = YOLOv2Net("train", CFG=YOLOv2Config)
if YOLOv2Config["GPU_NUMS"] > 0 :
    net = net.cuda()

darknet_weights = torch.load("utils/YOLOv2_pretrained_darknet.pth")
net.darknet.load_state_dict(darknet_weights)
net.conv.apply(weights_init)
net.conv1.apply(weights_init)
net.conv2.apply(weights_init)

optimizer = SGD(net.parameters(), lr=warm_lr, momentum=YOLOv2Config["MOMENTUM"], weight_decay=CFG["WEIGHT_DECAY"])

dataset = SSDDataSetForPytorch(transform=Augmentation(size=YOLOv2Config["IMAGE_SIZE"],mean=(0, 0, 0), scale=True),
                               target_transform=AnnotationTransform(classes=CFG["CLASSES"]))
data_loader = DataLoader(dataset, YOLOv2Config["BATCH_SIZE"],
                         shuffle=True, collate_fn=detection_collate, pin_memory=True)

bar = ProgressBar(YOLOv2Config["EPOCHS"], len(data_loader), "Loss:%.3f")
criterion = YoloLoss(CFG=YOLOv2Config)
for epoch in range(1, YOLOv2Config["EPOCHS"]):
    for i, (imgs, targets) in enumerate(data_loader):
        imgs = Variable(imgs.cuda() if YOLOv2Config["GPU_NUMS"] > 0 else imgs)
        targets = [anno.cuda() for anno in targets] if YOLOv2Config["GPU_NUMS"] > 0 else [anno for anno in targets]
        out = net(imgs)
        optimizer.zero_grad()
        conf_loss, class_loss, loc_loss = criterion(out, targets)
        loss = conf_loss + class_loss + loc_loss
        loss.backward()
        optimizer.step()

        bar.show(epoch, loss.item())

    torch.save(net.state_dict(), "YOLOv2_%3d.pth" % epoch)
    detection = YOLOv2Detection(CLASS=YOLOv2Config["CLASSES"], sourceImagePath="../testImages/demo.jpg", targetImagePath="", Net=net )
    detection.imageShow()