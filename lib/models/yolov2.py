import numpy as np
import torch
import math

from torch.nn import Module, Conv2d, MaxPool2d, BatchNorm2d, LeakyReLU, ModuleList, Sequential
from torch.nn.functional import sigmoid, softmax,mse_loss,cross_entropy

from lib.utils.utils_yolo import point_form, filter_box, nms, jaccard

from lib.utils.models import DetectionImage, BaseTransform

class YOLOv2Net(Module):
    def __init__(self, phase, CFG, eval=False):
        super(YOLOv2Net, self).__init__()
        self.CFG = CFG
        self.phase = phase

        self.darknet = ModuleList(self._get_darknet())
        self.anchors = np.array(self.CFG["ANCHORS"]).reshape(-1,2)
        self.class_num = self.CFG["CLASS_NUMS"]
        self.anchor_num = self.anchors.shape[0]
        self.feat_size = self.CFG["FEAT_SIZE"]
        self.priors = PriorBox(self.anchors, feat_size=self.feat_size)()

        self.conv2 = Sequential(
            ConvLayer(1024, 1024, 3, same_padding=True),
            ConvLayer(1024, 1024, 3, same_padding=True))
        self.conv1 = Sequential(
            ConvLayer(512, 64, 1, same_padding=True),
            ReorgLayer(2))
        self.conv = Sequential(
            ConvLayer(1280, 1024, 3, same_padding=True),
            Conv2d(1024, self.anchor_num * (self.class_num + 5), 1))

        if phase == 'test':
            self.detect = Detect(self.class_num, eval)
    def forward(self, x):
        for i in range(17):
            x = self.darknet[i](x)
        x1 = self.conv1(x)
        for i in range(17, len(self.darknet)):
            x = self.darknet[i](x)
        x2 = self.conv2(x)
        x = self.conv(torch.cat([x1, x2], 1))

        if self.CFG["GPU_NUMS"] > 0:
            self.priors = self.priors.cuda()
        b, c, h, w = x.size()
        if self.feat_size != h:
            self.priors = PriorBox(self.anchors, h)()
            self.feat_size = h
            self.priors = self.priors.cuda() if x.is_cuda else self.priors
        if self.priors.size(0) != b:
            self.priors = self.priors.repeat((b, 1, 1, 1))
        feat = x.permute(0, 2, 3, 1).contiguous().view(b, -1, self.anchor_num, self.class_num + 5)
        box_xy, box_wh = sigmoid(feat[..., :2]), feat[..., 2:4].exp()
        box_xy += self.priors[..., 0:2]
        box_wh *= self.priors[..., 2:]
        box_conf, box_prob = sigmoid(feat[..., 4:5]), feat[..., 5:]
        box_pred = torch.cat([box_xy, box_wh], 3) / h
        if self.phase == 'test':
            output = self.detect(box_pred, box_conf, softmax(box_prob, dim=3))
        else:
            output = (feat, box_pred, box_conf, box_prob)
        return output

    def _get_darknet(self):
        in_c, out_c = 3, 32
        flag = [1, 1, 0] * 3 + [1, 0, 1] * 2 + [0, 1, 0]
        pool = [0, 1, 4, 7, 13]
        size_flag = [3, 6, 9, 11, 14, 16]
        num = 18
        layers = []
        for i in range(num):
            ksize = 1 if i in size_flag else 3
            if i < 13:
                layers.append(ConvLayer(in_c, out_c, ksize, same_padding=True))
                layers.append(MaxPool2d(2)) if i in pool else None
            else:
                layers.append(MaxPool2d(2)) if i in pool else None
                layers.append(ConvLayer(in_c, out_c, ksize, same_padding=True))
            in_c, out_c = out_c, out_c * 2 if flag[i] else out_c // 2
        return layers

class ConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 if same_padding else 0

        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn   = BatchNorm2d(out_channels)
        self.relu = LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ReorgLayer(Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.size()
        s = self.stride
        x = x.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x = x.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        return x.view(B, s * s * C, H // s, W // s)

class PriorBox(object):
    def __init__(self, anchors, feat_size=13):
        super(PriorBox, self).__init__()
        self.feat_size = feat_size
        self.anchors = anchors
        self.anchor_num = anchors.shape[0]

    def __call__(self):
        x, y = np.meshgrid(np.arange(self.feat_size), np.arange(self.feat_size))
        x, y = x.repeat(self.anchor_num), y.repeat(self.anchor_num)
        xy = np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]
        wh = np.tile(self.anchors, (self.feat_size * self.feat_size, 1))
        output = torch.from_numpy(np.c_[xy, wh].astype(np.float32)).view(1, -1, self.anchor_num, 4)
        return output

class Detect(object):
    def __init__(self, class_num, eval=False, top_k=200):
        self.class_num = class_num
        self.top_k = top_k
        if eval:
            self.nms_t, self.score_t = self.CFG["EVAL_NMS_THRESHOLD"], self.CFG["EVAL_SCORE_THRESHOLD"]
        else:
            self.nms_t, self.score_t =self.CFG["NMS_THRESHOLD"], self.CFG["SCORE_THRESHOLD"]

    def __call__(self, loc, conf, prob):
        num = loc.size(0)
        loc = point_form(loc)
        output = torch.zeros(num, self.class_num, self.top_k, 5)
        for i in range(num):
            loc_t, score_t, label_t = filter_box(loc[i], conf[i], prob[i], self.score_t)
            for c in range(self.class_num):
                idx = label_t == c
                if idx.sum() == 0:
                    continue
                c_loc = loc_t[idx]
                c_score = score_t[idx]
                ids, count = nms(c_loc, c_score, self.nms_t, self.top_k)
                output[i, c, :count] = torch.cat((c_score[ids[:count]].unsqueeze(1), c_loc[ids[:count]]), 1)
        # sort the score --- note: flt shares same "memory" as output
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank > self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

class YoloLoss(Module):
    def __init__(self, CFG):
        super(YoloLoss, self).__init__()
        self.CFG = CFG
        self.anchors = torch.Tensor(CFG["ANCHORS"]).reshape(-1, 2)
        self.num_anchors = self.anchors.size(0)
        self.nclass = CFG["CLASS_NUMS"]
        self.no_object_scale = CFG["NO_OBJECT_SCALE"]
        self.object_scale = CFG["OBJECT_SCALE"]
        self.class_scale = CFG["CLASS_SCALE"]
        self.coord_scale = CFG["COORD_SCALE"]
        self.cuda_flag = True

    def forward(self, preds, targets, warm=False):
        feat, box_pred, box_conf, box_prob = preds
        num, f_size = feat.size(0), int(math.sqrt(feat.size(1)))
        box_match = torch.cat((sigmoid(feat[..., :2]), feat[..., 2:4]), -1)
        # TODO: simplify the mask
        coord_mask = torch.zeros_like(box_conf)
        conf_mask = torch.ones_like(box_conf)
        pos_mask = torch.zeros_like(box_conf)
        m_boxes = torch.zeros_like(box_conf).repeat(1, 1, 1, 5)
        if self.cuda_flag and feat.is_cuda:
            self.anchors = self.anchors.cuda()
        for idx in range(num):
            self._build_mask(box_pred[idx], targets[idx], self.anchors, f_size, coord_mask, conf_mask,
                       pos_mask, m_boxes, idx, self.object_scale, warm)
        loc_loss = mse_loss(coord_mask * box_match, coord_mask * m_boxes[..., :4],
                              size_average=False) / 2
        conf_loss = mse_loss(conf_mask * box_conf, conf_mask * pos_mask, size_average=False) / 2
        class_loss = cross_entropy(box_prob[pos_mask.byte().repeat(1, 1, 1, self.nclass)].view(-1, self.nclass),
                                     m_boxes[..., 4:5][pos_mask.byte()].view(-1).long(), size_average=False)
        return (conf_loss + class_loss + loc_loss) / num

    def _build_mask(self, box_pred, target, anchors, f_size, coord_mask, conf_mask, pos_mask, m_boxes, idx, scale, warm):
        box_pred = point_form(box_pred.view(-1, 4))
        overlap = jaccard(box_pred, target[:, :4])
        best_truth_overlap, best_truth_idx = overlap.max(1)
        # TODO: this is 0.6 in original paper
        conf_mask[idx][(best_truth_overlap > 0.5).view_as(conf_mask[idx])] = 0
        if warm:
            coord_mask[idx].fill_(1)
            m_boxes[idx, ..., 0:2] = 0.5
        t_xy = (target[:, :2] + target[:, 2:4]) * f_size / 2
        t_wh = (target[:, 2:4] - target[:, :2]) * f_size
        xy = torch.floor(t_xy).long()
        pos = xy[:, 1] * f_size + xy[:, 0]
        wh = t_wh / 2
        target_box = torch.cat((-wh, wh), dim=1)
        wh = anchors / 2
        anchor_box = torch.cat((-wh, wh), dim=1)
        overlap = jaccard(target_box, anchor_box)
        best_prior_overlap, best_prior_idx = overlap.max(1)
        coord_mask[idx, pos, best_prior_idx] = 1
        pos_mask[idx, pos, best_prior_idx] = 1
        conf_mask[idx, pos, best_prior_idx] = scale
        m_boxes[idx, pos, best_prior_idx] = torch.cat(
            (t_xy - xy.float(), torch.log(t_wh / anchors[best_prior_idx, :]), target[:, 4:5]), dim=1)

class YOLOv2Detection(DetectionImage):
    def __init__(self,CLASS, sourceImagePath, targetImagePath, Net):
        super(YOLOv2Detection, self).__init__(CLASS, sourceImagePath, targetImagePath, Net)
        self.scale = None

    def _transfer_image(self, image_orignal):
        h, w, _ = image_orignal.shape
        self.scale = np.array([w, h, w, h])
        transform = BaseTransform(size=416, mean=(0, 0, 0), scale=True)
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
    net = YOLOv2Net('train')
    net = net.eval()
    img = torch.randn((1, 3, 416, 416))
    out = net(img)
    print(out[0].size())