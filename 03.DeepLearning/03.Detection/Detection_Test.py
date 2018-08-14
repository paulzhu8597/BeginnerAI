import torch
from lib.Config import YOLOv2Config, YOLOv3Config
from lib.models.yolov2 import YOLOv2Net, YOLOv2Detection
from lib.models.yolov3 import YOLOv3Module, YOLOv3Detection

# cv2 => BGR
# PIL.Image => GRB

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

# model = YOLOv2Net(phase="test")
# model.load_state_dict(torch.load("utils/YOLOv2.pth"))
# detection = YOLOv2Detection(CLASS=YOLOv2Config["CLASSES"],
#                             sourceImagePath="../testImages/", targetImagePath="pre/", Net=model )
# detection.imageShow()

model = YOLOv3Module(cfg_file="utils/YOLOv3.cfg")
model.load_state_dict(torch.load("utils/YOLOv3_COCO.pth"))
detection = YOLOv3Detection(CLASS=load_classes("utils/coco.names"), sourceImagePath="../testImages/",
                            targetImagePath="pre/", Net=model)
detection.imageShow()

