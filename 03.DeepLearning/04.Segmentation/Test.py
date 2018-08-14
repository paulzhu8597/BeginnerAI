import torch
import fcn
import numpy as np
import skimage
fcn.datasets.VOC2011ClassSeg
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.models.fcn import FCN32sNet
from lib.dataset.pytorch_dataset import VOCClassSegDataSet
from lib.Config import FCNConfig
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


model = FCN32sNet()
model.load_state_dict(torch.load("FCN_ 67.pth"))
dataset = VOCClassSegDataSet(root='/input', year=2012, split="test", transform=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

visualizations = []
label_trues, label_preds = [], []

for batch_idx, (data, target) in tqdm(enumerate(data_loader),
                                           total=len(data_loader),
                                           ncols=80, leave=False):
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    score = model(data)

    imgs = data.data.cpu()
    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    lbl_true = target.data.cpu()
    for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
        img, lt = data_loader.dataset.untransform(img, lt)
        label_trues.append(lt)
        label_preds.append(lp)
        if len(visualizations) < 9:
            viz = fcn.utils.visualize_segmentation(
                lbl_pred=lp, lbl_true=lt, img=img, n_class=FCNConfig["CLASS_NUMS"],
                label_names=FCNConfig["CLASSES"])
            visualizations.append(viz)

metrics = label_accuracy_score(
    label_trues, label_preds, n_class=21)
metrics = np.array(metrics)
metrics *= 100
print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

viz = fcn.utils.get_tile_image(visualizations)
skimage.io.imsave('viz_evaluate.png', viz)
