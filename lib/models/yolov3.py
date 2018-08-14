import torch
import numpy as np
import random
from collections import defaultdict

from torch.nn import Sequential, ModuleList, Module, Conv2d, LeakyReLU, BatchNorm2d, Upsample, MSELoss, BCELoss
from torch.autograd import Variable
from skimage.transform import resize

from lib.utils.utils_yolo import build_targets
from lib.utils.models import DetectionImage
from lib.utils.utils_yolo import non_max_suppression

class YOLOv3Module(Module):
    def __init__(self, cfg_file):
        super(YOLOv3Module, self).__init__()
        self.cfg_file = cfg_file
        self.module_defs = self._parse_model_config()
        self.hyper_params, self.module_list = self._create_modules()
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses['recall'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def _parse_model_config(self):
        module_defs = []
        with open(self.cfg_file, 'r') as f:
            lines = f.read().split('\n')
            lines = [x for x in lines if x and not x.startswith('#')]
            lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
            for line in lines:
                if line.startswith('['): # This marks the start of a new block
                    module_defs.append({})
                    module_defs[-1]['type'] = line[1:-1].rstrip()
                    if module_defs[-1]['type'] == 'convolutional':
                        module_defs[-1]['batch_normalize'] = 0
                else:
                    key, value = line.split("=")
                    value = value.strip()
                    module_defs[-1][key.rstrip()] = value.strip()
        return module_defs
    def _create_modules(self):
        module_defs = self.module_defs
        hyperparams = module_defs.pop(0)
        output_filters = [int(hyperparams['channels'])]
        module_list = ModuleList()
        for i, module_def in enumerate(module_defs):
            modules = Sequential()

            if module_def['type'] == 'convolutional':
                bn = int(module_def['batch_normalize'])
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                modules.add_module('conv_%d' % i, Conv2d(in_channels=output_filters[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(module_def['stride']),
                                                            padding=pad,
                                                            bias=not bn))
                if bn:
                    modules.add_module('batch_norm_%d' % i, BatchNorm2d(filters))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky_%d' % i, LeakyReLU(0.1))

            elif module_def['type'] == 'upsample':
                upsample = Upsample( scale_factor=int(module_def['stride']),
                                        mode='nearest')
                modules.add_module('upsample_%d' % i, upsample)

            elif module_def['type'] == 'route':
                layers = [int(x) for x in module_def["layers"].split(',')]
                filters = sum([output_filters[layer_i] for layer_i in layers])
                modules.add_module('route_%d' % i, EmptyLayer())

            elif module_def['type'] == 'shortcut':
                filters = output_filters[int(module_def['from'])]
                modules.add_module("shortcut_%d" % i, EmptyLayer())

            elif module_def["type"] == "yolo":
                anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
                # Extract anchors
                anchors = [int(x) for x in module_def["anchors"].split(",")]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors = [anchors[i] for i in anchor_idxs]
                num_classes = int(module_def['classes'])
                img_height = int(hyperparams['height'])
                # Define detection layer
                yolo_layer = YOLOLayer(anchors, num_classes, img_height)
                modules.add_module('yolo_%d' % i, yolo_layer)
            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        return hyperparams, module_list

class EmptyLayer(Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = MSELoss()
        self.bce_loss = BCELoss()

    def forward(self, x, targets=None):
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  self.img_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,
                                                                                        targets.cpu().data,
                                                                                        scaled_anchors,
                                                                                        self.num_anchors,
                                                                                        self.num_classes,
                                                                                        g_dim,
                                                                                        self.ignore_thres,
                                                                                        self.img_dim)

            nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = Variable(mask.type(FloatTensor))
            cls_mask = Variable(mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).type(FloatTensor))
            conf_mask = Variable(conf_mask.type(FloatTensor))

            # Handle target variables
            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls.type(FloatTensor), requires_grad=False)

            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall

        else:
            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

class YOLOv3Detection(DetectionImage):
    def __init__(self,CLASS, sourceImagePath, targetImagePath, Net):
        super(YOLOv3Detection, self).__init__(CLASS, sourceImagePath, targetImagePath, Net)
        self.scale = None
        self.img_shape = (416,416)

    def _transfer_image(self, image_original):
        h, w, _ = image_original.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(image_original, pad, 'constant', constant_values=127.5) / 255.
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float().unsqueeze(0)
        return input_img

    def _draw_box(self, image_orignal, image_predict, filename):
        detections = non_max_suppression(image_predict, 80, 0.8, 0.4)[0]
        pad_x = max(image_orignal.shape[0] - image_orignal.shape[1], 0) * (416 / max(image_orignal.shape))
        pad_y = max(image_orignal.shape[1] - image_orignal.shape[0], 0) * (416 / max(image_orignal.shape))
        # Image height and width after padding is removed
        unpad_h = 416 - pad_y
        unpad_w = 416 - pad_x
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(self.colors, n_cls_preds)
        print ("Image: '%s'" % (filename))
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            print ('\t+ Label: %s, Conf: %.5f' % (self.CLASSES[int(cls_pred)], cls_conf.item()))
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * image_orignal.shape[0]
            box_w = ((x2 - x1) / unpad_w) * image_orignal.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * image_orignal.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * image_orignal.shape[1]

            x1,y1,box_w,box_h = x1.cpu().data.numpy(),y1.cpu().data.numpy(),box_w.cpu().data.numpy(),box_h.cpu().data.numpy()
            box = [x1, y1, x1 + box_w, y1 + box_h]
            label = '{} {:.2f}'.format(self.CLASSES[int(cls_pred)], cls_conf.item())
            self._draw_box_ex(image_orignal, label, box, i)

        return image_orignal

