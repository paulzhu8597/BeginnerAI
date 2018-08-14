# https://github.com/AceCoooool/detection-pytorch
import torch

from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from lib.dataset.pytorch_dataset import SSDDataSetForPytorch
from lib.models.ssd import SSDModule, MultiBoxLoss, SSDDetection
from lib.utils.models import Augmentation, AnnotationTransform
from lib.utils.functions import weights_init, detection_collate
from lib.ProgressBar import ProgressBar

IMAGE_CHANNEL = 3
IMAGE_SIZE = 300
GPU_NUMS = 0
BATCH_SIZE = 32
EPOCH = 120

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Net = SSDModule(phase="train")
if GPU_NUMS > 0:
    Net = Net.cuda()
Net.bone.load_state_dict(torch.load("vgg_rfc.pth"))
Net.extras.apply(weights_init)
Net.loc.apply(weights_init)
Net.conf.apply(weights_init)

optimizer = SGD(Net.parameters(), lr=1e-3,momentum=0.9, weight_decay=5e-4)

criterion = MultiBoxLoss().cuda() if GPU_NUMS > 0 else MultiBoxLoss()
Net.train()
loc_loss, conf_loss = 0, 0
dataset = SSDDataSetForPytorch(transform=Augmentation(), target_transform=AnnotationTransform(classes=VOC_CLASSES))
data_loader = DataLoader(dataset, BATCH_SIZE,
                         shuffle=True, collate_fn=detection_collate, pin_memory=True)
epoch_size = len(dataset) // BATCH_SIZE

bar = ProgressBar(EPOCH, len(data_loader), "Loss:%.3f")

for epoch in range(1,EPOCH):
    for i,(imgs, targets) in enumerate(data_loader):
        imgs = Variable(imgs.cuda() if GPU_NUMS > 0 else imgs)
        targets = [anno.cuda() for anno in targets] if GPU_NUMS > 0 else [anno for anno in targets]

        optimizer.zero_grad()

        out = Net(imgs)

        loss_l, loss_c = criterion(out, targets)
        loss = loss_c + loss_l
        loss.backward()
        optimizer.step()

        bar.show(epoch, loss.item())

    torch.save(Net.state_dict(), "SSD_%3d.pth" % epoch)
    predictImage = SSDDetection(VOC_CLASSES, sourceImagePath="../testImages/demo.jpg", targetImagePath=".", Net=Net)
    predictImage.imageShow()


