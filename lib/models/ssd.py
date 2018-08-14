import torch
import numpy as np

from math import sqrt

from torch.nn import Module, MaxPool2d, Conv2d, ReLU, ModuleList,Parameter
from torch.nn.init import constant_
from torch.nn.functional import softmax, relu, smooth_l1_loss,cross_entropy
from lib.utils.utils_ssd import decode, nms, match, log_sum_exp
from lib.utils.models import DetectionImage, BaseTransform

class SSDModule(Module):
    def __init__(self, phase="train", image_size=(300,300,3), num_classes=21):
        super(SSDModule, self).__init__()
        self.IMAGE_CHANNEL = image_size[2]
        self.IMAGE_SIZE = image_size[0]
        self._vggfeat_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        self._extras_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
        self._mbox_cfg = [(512, 4), (1024, 6), (512, 6), (256, 6), (256, 4), (256, 4)]
        self.config = {
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_dim': 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'v2',
        }
        self.priors = PriorBox(self.config)()
        self.num_classes = num_classes
        self.phase = phase
        self.l = 23

        base_ = self._vggFeat()
        fc7_, extras = self._extras()
        loc_, conf_ = self._multibox()
        self.bone = ModuleList(base_ + fc7_)
        self.l2norm = L2Norm(512, 20)
        self.extras = ModuleList(extras)
        self.loc = ModuleList(loc_)
        self.conf = ModuleList(conf_)

        if self.phase == 'test':
            self.detect = Detect(num_classes, 200, 0.01, 0.45, self.config)

    def forward(self, x):
        source, loc, conf = list(), list(), list()
        for k in range(self.l):
            x = self.bone[k](x)
        source.append(self.l2norm(x))
        for k in range(self.l, len(self.bone)):
            x = self.bone[k](x)
        source.append(x)
        for k, v in enumerate(self.extras):
            x = relu(v(x), inplace=True)
            if k % 2 == 1:
                source.append(x)
        # apply multibox head to source layers
        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if not self.priors.is_cuda and loc.is_cuda:
            self.priors = self.priors.cuda()
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                softmax(conf.view(conf.size(0), -1, self.num_classes), dim=2),
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def _vggFeat(self):
        layers = []
        in_channels = self.IMAGE_CHANNEL
        for v in self._vggfeat_cfg:
            if v == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, ReLU(inplace=True)]
                in_channels = v
        return layers

    def _extras(self):
        fc7 = [
            MaxPool2d(3, 1, 1),
            Conv2d(512, 1024, 3, 1, 6, 6),
            ReLU(inplace=True),
            Conv2d(1024, 1024, 1),
            ReLU(inplace=True)]
        layers = []
        in_channels = 1024
        flag = False
        for k, v in enumerate(self._extras_cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [
                        Conv2d(in_channels, self._extras_cfg[k + 1],kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return fc7, layers

    def _multibox(self):
        loc_layers = []
        conf_layers = []
        for channel, n in self._mbox_cfg:
            loc_layers += [Conv2d(channel, n * 4, 3, 1, 1)]
            conf_layers += [Conv2d(channel, n * self.num_classes, 3, 1, 1)]
        return loc_layers, conf_layers

class L2Norm(Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = Parameter(torch.randn(self.n_channels))  # only Parameter can be "check"
        self.reset_parameters()

    def reset_parameters(self):
        constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class Detect(object):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh, cfg):
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def __call__(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors, self.num_classes)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i]
            # each class:
            # step1---delete score<conf_thresh
            # step2---non-maximum suppression
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[:, cl].gt(self.conf_thresh)
                scores = conf_scores[:, cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        # sort the score --- note: flt shares same "memory" as output
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank > self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

    def __call__(self):
        output = None
        for k, f in enumerate(self.feature_maps):
            mean = None
            f_k = self.image_size / self.steps[k]
            cx, cy = np.meshgrid(np.arange(0, f) + 0.5, np.arange(0, f) + 0.5)
            cy, cx = cy / f_k, cx / f_k
            s_k = self.min_sizes[k] / self.image_size
            s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
            wh = np.tile(s_k, (f, f))
            temp = np.vstack((cx.ravel(), cy.ravel(), wh.ravel(), wh.ravel())).transpose()
            mean = temp if mean is None else np.c_[mean, temp]
            wh = np.tile(s_k_prime, (f, f))
            temp = np.vstack((cx.ravel(), cy.ravel(), wh.ravel(), wh.ravel())).transpose()
            mean = np.c_[mean, temp]
            for ar in self.aspect_ratios[k]:
                w = np.tile(s_k * sqrt(ar), (f, f))
                h = np.tile(s_k / sqrt(ar), (f, f))
                temp = np.vstack((cx.ravel(), cy.ravel(), w.ravel(), h.ravel())).transpose()
                mean = np.c_[mean, temp]
                temp = np.vstack((cx.ravel(), cy.ravel(), h.ravel(), w.ravel())).transpose()
                mean = np.c_[mean, temp]
            output = mean.reshape((-1, 4)) if output is None else np.r_[output, mean.reshape((-1, 4))]
        output = torch.from_numpy(output.astype(np.float32))
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class MultiBoxLoss(Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = 21
        self.threshold = 0.5
        self.negpos_ratio = 3
        self.variance = [0.1,0.2]

    def forward(self, preds, targets):
        loc_data, conf_data, priors = preds
        num = loc_data.size(0)
        num_priors = priors.size(0)
        # match priors (priors->nearest target)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if loc_data.is_cuda:
            loc_t, conf_t = loc_t.cuda(), conf_t.cuda()
        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        pos = conf_t > 0
        # location loss
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = smooth_l1_loss(loc_p, loc_t, size_average=False)

        # evaluate each priors's loss (the same as the paper)
        batch_conf = conf_data
        loss_c = (log_sum_exp(batch_conf) - batch_conf.gather(2, conf_t.unsqueeze(2))).squeeze(2)
        # hard negative mining: note: the batch size of each iteration is not the same
        # find the "max loss" background
        loss_c[pos] = 0  # filter out pos boxes
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # size: [num, 1]
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # confidence loss (pos:neg=1:3)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weightd = conf_t[(pos + neg).gt(0)]
        loss_c = cross_entropy(conf_p, targets_weightd, size_average=False)

        return loss_l / num_pos.sum().type(torch.FloatTensor), loss_c / num_pos.sum().type(torch.FloatTensor)

class SSDDetection(DetectionImage):
    def __init__(self,CLASS, sourceImagePath, targetImagePath, Net):
        super(SSDDetection, self).__init__(CLASS, sourceImagePath, targetImagePath, Net)
        self.scale = None

    def _transfer_image(self, image_orignal):
        h, w, _ = image_orignal.shape
        self.scale = np.array([w, h, w, h])
        transform = BaseTransform()
        x, _, _ = transform(image_orignal)
        x = x[:, :, (2, 1, 0)]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x

    def _draw_box(self, image_orignal, image_predict, filename):
        for i in range(image_predict.size(1)):
            idx = (image_predict[0, i, :, 0] > 0.5)
            dets = image_predict[0, i][idx].view(-1, 5)
            if dets.numel() == 0:
                continue
            print('Find {} {} for {}.'.format(dets.size(0), self.CLASSES[i], filename))
            score, loc = dets[:, 0], dets[:, 1:].cpu().numpy() * self.scale
            for k in range(len(score)):
                label = '{} {:.2f}'.format(self.CLASSES[i], score[k])
                self._draw_box_ex(image_orignal, label, loc[k], i)
        return image_orignal

if __name__ == '__main__':
    net = SSDModule(phase='train')
    img = torch.randn((1, 3, 300, 300))
    out = net(img)
    print(out)