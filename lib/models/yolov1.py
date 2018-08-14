import torch
import cv2
import numpy as np

from torch.nn import Module, ZeroPad2d, Conv2d, MaxPool2d,Linear, Dropout, LeakyReLU, Sigmoid
from torch.nn.functional import mse_loss
from torch.autograd import Variable

from lib.utils.models import DetectionImage
class YOLOv1Net(Module):
    def __init__(self, CFG):
        super(YOLOv1Net, self).__init__()
        self.CFG = CFG
        self.layer1 = ZeroPad2d(padding=3) # 454,454,3
        self.layer2 = Conv2d(in_channels=CFG["IMAGE_CHANNEL"], out_channels=64, kernel_size=7, stride=2, padding=0)# 224,224,64

        self.layer3 = MaxPool2d(kernel_size=2, padding=0)# 112,112,64
        self.layer4 = Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1) # 112,112,192
        self.layer5 = MaxPool2d(kernel_size=2) # 56,56,192

        self.layer6 = Conv2d(in_channels=192, out_channels=128, kernel_size=1) # 56,56,128
        self.layer7 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # 56,56,256
        self.layer8 = Conv2d(in_channels=256, out_channels=256, kernel_size=1) # 56,56,256
        self.layer9 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) # 56,56,512
        self.layer10 = MaxPool2d(kernel_size=2) # 28,28,512

        self.layer11 = Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.layer12 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.layer13 = Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.layer14 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.layer15 = Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.layer16 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.layer17 = Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.layer18 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.layer19 = Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.layer20 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1) # 28,28,1024

        self.layer21 = MaxPool2d(kernel_size=2) # 14,14,1024

        self.layer22 = Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.layer23 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.layer24 = Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.layer25 = Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)

        self.layer26 = Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1) # 14,14,1024

        self.layer27 = ZeroPad2d(padding=1) # 16,16,1024

        self.layer28 = Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2) # 7,7,1024
        self.layer29 = Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.layer30 = Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1) # 7,7,1024

        self.layer33 = Linear(in_features=1024 * 7 * 7, out_features=512)
        self.layer34 = Linear(in_features=512, out_features=4096)

        self.layer35 = Dropout(p=0.5)
        self.layer36 = Linear(in_features=4096, out_features=CFG["CELL_NUMS"] * CFG["CELL_NUMS"] * (5 * CFG["BOXES_EACH_CELL"] + CFG["CLASS_NUMS"]))

    def forward(self, x):
        ALPHA = self.CFG["ALPHA"]
        output = self.layer1(x)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer2(output))
        output = self.layer3(output)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer4(output))
        output = self.layer5(output)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer6(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer7(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer8(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer9(output))
        output = self.layer10(output)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer11(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer12(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer13(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer14(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer15(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer16(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer17(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer18(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer19(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer20(output))
        output = self.layer21(output)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer22(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer23(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer24(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer25(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer26(output))
        output = self.layer27(output)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer28(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer29(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer30(output))
        output = output.view(output.size(0), -1)

        output = LeakyReLU(negative_slope=ALPHA)(self.layer33(output))
        output = LeakyReLU(negative_slope=ALPHA)(self.layer34(output))
        output = self.layer35(output)
        output = LeakyReLU(negative_slope=ALPHA)(self.layer36(output))
        output = Sigmoid()(output)
        output = output.view(-1, self.CFG["CELL_NUMS"], self.CFG["CELL_NUMS"], 5 * self.CFG["BOXES_EACH_CELL"] + self.CFG["CLASS_NUMS"])
        return output

class YOLOv1Loss(Module):
    '''
    定义一个torch.nn中并未实现的网络层，以使得代码更加模块化
    torch.nn.Modules相当于是对网络某种层的封装，包括网络结构以及网络参数，和其他有用的操作如输出参数
    继承Modules类，需实现__init__()方法，以及forward()方法
    '''
    def __init__(self):
        super(YOLOv1Loss,self).__init__()
        self.S = self.CFG["CELL_NUMS"]    #7代表将图像分为7x7的网格
        self.B = self.CFG["BOXES_EACH_CELL"]    #2代表一个网格预测两个框
        self.l_coord = self.CFG["L_COORD"]   #5代表 λcoord  更重视8维的坐标预测
        self.l_noobj = self.CFG["L_NOOBJ"]   #0.5代表没有object的bbox的confidence loss
        self.gpu_nums = self.CFG["GPU_NUMS"]
    def compute_iou(self, box1, box2):
        '''
        计算两个框的重叠率IOU
        通过两组框的联合计算交集，每个框为[x1，y1，x2，y2]。
        Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        # wh(wh<0)= 0  # clip at 0
        wh= (wh < 0).float()
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c](1470=7*7*30)
        target_tensor: (tensor) size(batchsize,S,S,30)

        Mr.Li个人见解：
        本来有，预测无--》计算response loss响应损失
        本来有，预测有--》计算not response loss 未响应损失
        本来无，预测无--》无损失(不计算)
        本来无，预测有--》计算不包含obj损失  只计算第4,9位的有无物体概率的loss
        '''
        # N为batchsize
        N = pred_tensor.size()[0]
        # 坐标mask    4：是物体或者背景的confidence    >0 拿到有物体的记录
        coo_mask = target_tensor[:,:,:,4] > 0
        # 没有物体mask                                 ==0  拿到无物体的记录
        noo_mask = target_tensor[:,:,:,4] == 0
        # unsqueeze(-1) 扩展最后一维，用0填充，使得形状与target_tensor一样
        # coo_mask、noo_mask形状扩充到[?,7,7,30]
        # coo_mask 大部分为0   记录为1代表真实有物体的网格
        # noo_mask  大部分为1  记录为1代表真实无物体的网格
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # coo_pred 取出预测结果中有物体的网格，并改变形状为（xxx,30）  xxx代表一个batch的图片上的存在物体的网格总数
        # 30代表2*5+20   例如：coo_pred[72,30]
        coo_pred = pred_tensor[coo_mask].view(-1,30)
        # 一个网格预测的两个box  30的前10即为2个x,y,w,h,c，并调整为（xxx,5） xxx为所有真实存在物体的预测框，而非所有真实存在物体的网格     例如：box_pred[144,5]
        # contiguous将不连续的数组调整为连续的数组
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        # #[x2,y2,w2,h2,c2]
        # 每个网格预测的类别  后20
        class_pred = coo_pred[:,10:]

        # 对真实标签做同样操作
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # 计算不包含obj损失  即本来无，预测有
        # 在预测结果中拿到真实无物体的网格，并改变形状为（xxx,30）  xxx代表一个batch的图片上的不存在物体的网格总数    30代表2*5+20   例如：[1496,30]
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)      # 例如：[1496,30]
        # ByteTensor：8-bit integer (unsigned)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()) if self.gpu_nums > 0 else torch.ByteTensor(noo_pred.size())   # 例如：[1496,30]
        noo_pred_mask.zero_()   #初始化全为0
        # 将第4、9  即有物体的confidence置为1
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        # 拿到第4列和第9列里面的值（即拿到真实无物体的网格中，网络预测这些网格有物体的概率值）    一行有两个值（第4和第9位）                           例如noo_pred_c：2992        noo_target_c：2992
        # noo pred只需要计算类别c的损失
        noo_pred_c = noo_pred[noo_pred_mask]
        # 拿到第4列和第9列里面的值  真值为0，真实无物体（即拿到真实无物体的网格中，这些网格有物体的概率值，为0）
        noo_target_c = noo_target[noo_pred_mask]
        # 均方误差    如果 size_average = True，返回 loss.mean()。    例如noo_pred_c：2992        noo_target_c：2992
        # nooobj_loss 一个标量
        nooobj_loss = mse_loss(noo_pred_c,noo_target_c,size_average=False)


        #计算包含obj损失  即本来有，预测有  和  本来有，预测无
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()) if self.gpu_nums > 0 else torch.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()) if self.gpu_nums > 0 else torch.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        # 选择最好的IOU
        for i in range(0,box_target.size()[0],2):
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2] -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2] +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2] -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2] +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda() if self.gpu_nums > 0 else max_index.data
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1
        # 1.response loss响应损失，即本来有，预测有   有相应 坐标预测的loss  （x,y,w开方，h开方）参考论文loss公式
        # box_pred [144,5]   coo_response_mask[144,5]   box_pred_response:[72,5]
        # 选择IOU最好的box来进行调整  负责检测出某物体
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        # box_pred_response:[72,5]     计算预测 有物体的概率误差，返回一个数
        contain_loss = mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        # 计算（x,y,w开方，h开方）参考论文loss公式
        loc_loss = mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + \
                   mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)

        # 2.not response loss 未响应损失，即本来有，预测无   未响应
        # box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        # box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        # box_target_not_response[:,4]= 0
        # box_pred_response:[72,5]
        # 计算c  有物体的概率的loss
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        # 3.class loss  计算传入的真实有物体的网格  分类的类别损失
        class_loss = mse_loss(class_pred,class_target,size_average=False)
        # 除以N  即平均一张图的总损失
        return (self.l_coord*loc_loss +
                contain_loss +
                # not_contain_loss +
                self.l_noobj*nooobj_loss +
                class_loss)/N

class YOLOv1Detection(DetectionImage):
    def __init__(self, CLASSES, sourceImagePath, targetImagePath, Net):
        super(YOLOv1Detection, self).__init__(CLASSES, sourceImagePath, targetImagePath, Net)
        self.IMAGE_CHANNEL = 3
        self.IMAGE_SIZE = 448

    def _transfer_image(self, image_orignal):
        img = cv2.resize(image_orignal, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))
        return inputs

    def _draw_box(self,image_orignal, image_predict, filename):
        pred_image = image_predict.data.cpu().numpy()
        pred_image = np.reshape(pred_image, newshape=(-1, 7*7*30))
        results = []
        for i in range(pred_image.shape[0]):
            results.append(self._interpret_output(pred_image[i]))

        results = results[0]
        img_h, img_w = image_orignal.shape
        for i in range(len(results)):
            results[i][1] *= 1.0 * img_w / 448
            results[i][2] *= 1.0 * img_h / 448
            results[i][3] *= 1.0 * img_w / 448
            results[i][4] *= 1.0 * img_h / 448
        img = cv2.resize(image_orignal, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        self._draw_result(img, results)

        return img

    def _interpret_output(self, output):
        # 7,7,30
        cell_size = 7
        num_class = 20
        boxes_per_cell = 2
        boundary1 = cell_size * cell_size * num_class
        boundary2 = boundary1 + cell_size * cell_size * boxes_per_cell

        CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor']

        probs = np.zeros((cell_size, cell_size,
                          boxes_per_cell, num_class))
        class_probs = np.reshape(output[0:boundary1],(cell_size, cell_size, num_class))
        scales      = np.reshape(output[boundary1:boundary2],(cell_size, cell_size, boxes_per_cell))
        boxes = np.reshape(output[boundary2:],(cell_size, cell_size, boxes_per_cell, 4))
        offset = np.array([np.arange(cell_size)] * cell_size * boxes_per_cell)
        offset = np.transpose(np.reshape(offset,[boxes_per_cell, cell_size, cell_size]),(1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= 448

        for i in range(boxes_per_cell):
            for j in range(num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= .2, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self._iou(boxes_filtered[i], boxes_filtered[j]) > .5:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [CLASSES[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def _iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def _draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)
