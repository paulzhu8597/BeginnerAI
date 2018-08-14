import torch

from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from lib.dataset.pytorch_dataset import ListDataset
from lib.models.yolov3 import YOLOv3Module, YOLOv3Detection
from lib.utils.models import AnnotationTransform
from lib.utils.functions import weights_init_normal
from lib.ProgressBar import ProgressBar

YOLOv3Config = {
    "CLASS_NUMS" : 20,
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "IMAGE_SIZE" : 416,
    "EPOCHS" : 120,
    "IMAGE_CHANNEL" : 3,
    "GPU_NUMS" : 0,
    "LEARNING_RATE" : 1e-4,
    "BATCH_SIZE" : 1,
    "MOMENTUM" : 0.9,
    "WEIGHT_DECAY" : 1e-4,
    "DATA_PATH" : "/input/"
}
warm_lr = YOLOv3Config["LEARNING_RATE"] / YOLOv3Config["BATCH_SIZE"]

net = YOLOv3Module("utils/YOLOv3.cfg")
net.apply(weights_init_normal)

if YOLOv3Config["GPU_NUMS"] > 0 :
    net = net.cuda()

net.train()

optimizer = SGD(net.parameters(), lr=warm_lr, momentum=YOLOv3Config["MOMENTUM"],
                weight_decay=YOLOv3Config["WEIGHT_DECAY"])

data_loader = DataLoader(
    ListDataset("/input", target_transform=AnnotationTransform(classes=YOLOv3Config["CLASSES"])),
    batch_size=YOLOv3Config["BATCH_SIZE"], shuffle=False)

bar = ProgressBar(YOLOv3Config["EPOCHS"], len(data_loader), "Loss:%.3f")

for epoch in range(1, YOLOv3Config["EPOCHS"]):
    for i, (imgs, targets) in enumerate(data_loader):
        imgs = Variable(imgs.cuda() if YOLOv3Config["GPU_NUMS"] > 0 else imgs)
        # targets = [anno.cuda() for anno in targets] if CFG["GPU_NUMS"] > 0 else [anno for anno in targets]

        optimizer.zero_grad()
        loss = net(imgs, targets)

        loss.backward()
        optimizer.step()

        bar.show(epoch, loss.item())

    torch.save(net.state_dict(), "YOLOv3_%3d.pth" % epoch)
    detection = YOLOv3Detection(CLASS=YOLOv3Config["CLASSES"], sourceImagePath="../testImages/",
                                targetImagePath="pre/", Net=net)
    detection.imageShow()