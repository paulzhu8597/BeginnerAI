# coding=utf-8
import random
import os
import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET

from os.path import join
from os import listdir
from PIL import Image
from skimage.transform import resize
from scipy.misc import imread, imresize

from torch.utils.data import Dataset
from torchvision.transforms.transforms import ToTensor, Normalize, Compose

from lib.dataset.BasicDataSet import Cifar10DataSet,MnistDataSet,STLDataSet
from lib.utils.functions import is_image_file, load_img

class DataSetFromFolderForPix2Pix(Dataset):
    def __init__(self, image_dir):
        super(DataSetFromFolderForPix2Pix, self).__init__()
        self.photo_path = join(image_dir, "A")
        self.sketch_path = join(image_dir, "B")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x) ]

        transform_list = [ToTensor(),
                          Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        self.transform = Compose(transform_list)

    def __getitem__(self, index):
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

class DataSetFromFolderForCycleGAN(Dataset):
    def __init__(self, image_dir, subfolder='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DataSetFromFolderForCycleGAN, self).__init__()
        self.input_path = join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(listdir(self.input_path))]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        img_fn = join(self.input_path, self.image_filenames[index])
        img = load_img(img_fn)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)

class Cifar10DataSetForPytorch(Dataset):
    def __init__(self, root="/input/cifar10/", train=True,transform=None, target_transform=None, target_label=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        reader = Cifar10DataSet(self.root, special_label=target_label)
        (self.train_data, self.train_label), (self.test_data, self.test_label) = reader.read(channel_first=False)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class MNISTDataSetForPytorch(Dataset):
    def __init__(self, root="/input/mnist.npz", train=True, radio=0.9, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        reader = MnistDataSet(root=self.root, radio=radio)

        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = reader.read()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class STLDataSetForPytorch(Dataset):
    def __init__(self, root="/input/STLB/", train=True,transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        reader = STLDataSet(self.root)
        (self.train_data, self.train_label), (self.test_data, self.test_label) = reader.read(channel_first=False)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class yoloDataset(Dataset):

    def __init__(self,root,list_file,train,transform):
        self.image_size = 448
        print('数据初始化')
        self.root=root
        self.train = train
        self.transform=transform    #对图像转化
        self.fnames = []            #图像名字
        self.boxes = []
        self.labels = []
        self.mean = (123,117,104)   #RGB均值

        with open(list_file) as f:
            lines  = f.readlines()

        # 遍历voc2012train.txt每一行
        for line in lines:
            splited = line.strip().split()
            # 赋值图像名字
            self.fnames.append(splited[0])
            # 赋值一张图的物体总数
            num_faces = int(splited[1])
            box=[]
            label=[]
            # 遍历一张图的所有物体
            #  bbox坐标（4个值）   物体对应的类的序号（1个值）  所以要加5*i
            for i in range(num_faces):
                x = float(splited[2+5*i])
                y = float(splited[3+5*i])
                x2 = float(splited[4+5*i])
                y2 = float(splited[5+5*i])
                c = splited[6+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            # bbox  写入所有物体的坐标值
            self.boxes.append(torch.Tensor(box))
            # label 写入标签
            self.labels.append(torch.LongTensor(label))
        # 数据集中图像总数
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        '''
        继承Dataset，需实现该方法，得到一个item
        '''
        fname = self.fnames[idx]
        # 读取图像
        img = cv2.imread(os.path.join(self.root+fname))
        # clone 深复制，不共享内存
        # 拿出对应的bbox及 标签对应的序号
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # 如果为训练集,进行数据增强
        if self.train:
            # 随机翻转
            img, boxes = self.random_flip(img, boxes)
            #固定住高度，以0.6-1.4伸缩宽度，做图像形变
            img,boxes = self.randomScale(img,boxes)
            # 随机模糊
            img = self.randomBlur(img)
            # 随机亮度
            img = self.RandomBrightness(img)
            # 随机色调
            img = self.RandomHue(img)
            # 随机饱和度
            img = self.RandomSaturation(img)
            # 随机转换
            img,boxes,labels = self.randomShift(img,boxes,labels)

        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.BGR2RGB(img) #因为pytorch自身提供的预训练好的模型期望的输入是RGB
        img = self.subMean(img,self.mean) #减去均值
        img = cv2.resize(img,(self.image_size,self.image_size))   #改变形状到（224,224）

        '''
        这里对img进行了resize，那按道理来说也应该多boxes做resize，因为img相当于做了缩放，boxes也应该按照比例进行
        缩放才对，这里为什么没有做缩放呢，这是因为boxes做了归一化，相当于boxes的值不是一个绝对坐标值，而是boxes与
        原图的比率值，这样不管img做怎样的缩放，boxes的比例是不变的
        '''

        # 拿到图像对应的真值，以便计算loss
        target = self.encoder(boxes,labels)# 7x7x30    一张图被分为7x7的网格;30=（2x5+20） 一个网格预测两个框   一个网格预测所有分类概率，VOC数据集分类为20类
        # 图像转化
        for t in self.transform:
            img = t(img)
        #返回 最终处理好的img 以及 对应的 真值target（形状为网络的输出结果的大小）
        return img,target

    def __len__(self):
        '''
        继承Dataset，需实现该方法，得到数据集中图像总数
        '''
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[x1,y1,x2,y2],[]] 在上面已经将boxes归一化了(除以了整个图片的尺寸)
        labels (tensor) [...]
        return 7x7x30
        '''
        target = torch.zeros((7,7,30))
        cell_size = 1./7
        # boxes[:, 2:]代表  2: 代表xmax,ymax
        # boxes[:, :2]代表  ：2  代表xmin,ymin
        # wh代表  bbox的宽（xmax-xmin）和高（ymax-ymin）
        wh = boxes[:,2:]-boxes[:,:2]
        # bbox的中心点坐标
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        # cxcy.size()[0]代表 一张图像的物体总数
        # 遍历一张图像的物体总数
        for i in range(cxcy.size()[0]):
            # 拿到第i行数据，即第i个bbox的中心点坐标（相对于整张图，取值在0-1之间）
            cxcy_sample = cxcy[i]
            # ceil返回数字的上入整数
            # cxcy_sample为一个物体的中心点坐标，求该坐标位于7x7网格的哪个网格
            # cxcy_sample坐标在0-1之间  现在求它再0-7之间的值，故乘以7
            # ij长度为2，代表7x7框的某一个框 负责预测一个物体
            ij = (cxcy_sample/cell_size).ceil()-1
            # 每行的第4和第9的值设置为1，即每个网格提供的两个真实候选框 框住物体的概率是1.
            #xml中坐标理解：原图像左上角为原点，右边为x轴，下边为y轴。
            # 而二维矩阵（x，y）  x代表第几行，y代表第几列
            # 假设ij为（1,2） 代表x轴方向长度为1，y轴方向长度为2
            # 二维矩阵取（2,1） 从0开始，代表第2行，第1列的值
            # 画一下图就明白了
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            # 加9是因为前0-9为两个真实候选款的值。后10-20为20分类   将对应分类标为1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            # 匹配到的网格的左上角的坐标（取值在0-1之间）（原作者）
            # 根据二维矩阵的性质，从上到下  从左到右
            xy = ij*cell_size
            #cxcy_sample：第i个bbox的中心点坐标     xy：匹配到的网格的左上角相对坐标
            # delta_xy：真实框的中心点坐标相对于  位于该中心点所在网格的左上角   的相对坐标，此时可以将网格的左上角看做原点，你这点相对于原点的位置。取值在0-1，但是比1/7小
            delta_xy = (cxcy_sample -xy)/cell_size
            # x,y代表了检测框中心相对于网格边框的坐标。w,h的取值相对于整幅图像的尺寸
            # 写入一个网格对应两个框的x,y,   wh：bbox的宽（xmax-xmin）和高（ymax-ymin）（取值在0-1之间）
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        '''
         随机模糊
        '''
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.6-1.4伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.6,1.4)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        '''
        随机翻转
        '''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

class YoLoV2DataSetForPytorch(Dataset):
    def __init__(self, filepath, transform, anchors, image_size=416):
        self.ANCHOR_VALUE = anchors
        self.BOUX_CELL_SIZE = 13
        self.filepath = filepath
        self.image_size = image_size
        self.transform = transform
        self.frames = []
        self.boxes = []
        self.labels = []
        with open(self.filepath) as f:
            lines  = f.readlines()

        for line in lines:
            info = line.strip().split(" ")
            filename = info[0]
            self.frames.append(filename)
            num_box = info[1]
            boxes = []
            labels = []
            for box_index in range(int(num_box)):
                box = info[box_index + 2]
                box_pt = box.split(",")
                boxes.append([float(x) for x in box_pt[:4]])
                labels.append(int(box_pt[4:][0]))
            self.boxes.append(boxes)
            self.labels.append(labels)
    '''
    这里返回的是两个值
    image : 大小为IMAGE_SIZE的图片信息，IMAGE_SIZE为416
    target : 13*13*5*(5+20)] = 13*13*5*25 - For VOC
    相当于13 * 13个格子，每个格子包含5个anchors box的信息，每个anchors box包含25位信息
    0-3，表示4个坐标,(x,y,w,h)
    4，表示置信度(是否使用这个anchor来预测物体)
    5-24，表示类别的one-hot编码
    '''
    def __getitem__(self, index):
        image  = Image.open(self.frames[index])
        current_image_size = [image.width, image.height]
        target = self.convert_ground_truth(self.boxes[index], self.labels[index], current_image_size)

        image =  image.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        image = np.array(image, dtype='float32')
        image /= 255

        for t in self.transform:
            image = t(image)

        return image, target

    def __len__(self):
        return len(self.frames)

    def convert_ground_truth(self, boxes, labels, current_image_size):
        anchors_length = len(self.ANCHOR_VALUE)
        half = self.ANCHOR_VALUE / 2.
        half = np.asarray(half, dtype='float32')
        anchors_min = -half
        anchors_max = half
        anchors_areas = half[:,1]*half[:,0]*4
        width, height = current_image_size

        target = np.zeros((self.BOUX_CELL_SIZE, self.BOUX_CELL_SIZE, anchors_length, 25))

        # object_mask  = np.zeros((self.BOUX_CELL_SIZE, self.BOUX_CELL_SIZE, anchors_length, 1))
        # object_value = np.zeros((self.BOUX_CELL_SIZE, self.BOUX_CELL_SIZE, anchors_length, 5))

        '''
        width/cfg.BOUX_CELL_SIZE : 每个单元格的宽度
        height/cfg.BOUX_CELL_SIZE ： 每个单元格的长度
        那box_wh就很容易理解了，就是grouth box在13*13的特征图上，宽和高占多少个单元格
        '''
        Cell_W = width/self.BOUX_CELL_SIZE
        Cell_H = height/self.BOUX_CELL_SIZE
        for box, label in zip(boxes, labels):
            '''
            
            '''
            box_wh = box[2:4]/np.array([Cell_W, Cell_H])
            half = box_wh / 2
            box_half = np.repeat(np.asarray(half, dtype='float32').reshape((1,2)), anchors_length, axis=0)
            box_min = -box_half
            box_max = box_half
            intersect_min = np.maximum(box_min, anchors_min)
            intersect_max = np.minimum(box_max, anchors_max)
            intersect_box = np.maximum(intersect_max-intersect_min, 0.)
            intersect_areas = intersect_box[:, 0]*intersect_box[:, 1]
            box_areas = box_half[:,0]*box_half[:,1]*4
            iou = intersect_areas/(box_areas+anchors_areas-intersect_areas)
            '''
            IOU = 相交的面积 / (面积总和-相交的面积)
            '''
            maximum_iou = np.max(iou)
            if maximum_iou>0:
                index = np.argmax(iou) # index表示将来负责预测的那个anchor的index
                x = (box[0]+box[2]/2)/float(width) # box中心点x坐标，归一化之后的
                y = (box[1]+box[3]/2)/float(height) # box中心点y坐标，归一化之后的

                w = box[2]/float(width) # box的宽度，归一化之后的
                h = box[3]/float(height) # box的长度，归一化之后的

                x_index = np.int((box[0]+box[2]/2)/(width/self.BOUX_CELL_SIZE))
                y_index = np.int((box[1]+box[3]/2)/(height/self.BOUX_CELL_SIZE))

                '''
                这里主要是要确定哪一个anchor box来负责这个grouth box的坐标预测,在做损失的时候就可以找到对应的anchor box坐标进行微调。
                至于如何确定哪一个anchor，就是根据IOU
                '''
                target[x_index, y_index, index, 4] = 1
                target[x_index, y_index, index, 0:4] = [x,y,w,h] # 这四个之都是归一化的值，相当于box各个坐标在原图中的比例，所以无论img怎么resize，box的比例是不会变的。
                target[x_index, y_index, index, 5 + label] = 1

        return target

class SSDDataSetForPytorch(Dataset):
    def __init__(self, root='/input/', image_set=[('2012', 'trainval')], transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_set:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, item):
        img, gt, h, w = self.pull_item(item)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channel = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]  # bgr->rgb
            target = np.c_[boxes, np.expand_dims(labels, axis=1)]

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        # Note: here use the bgr form (rgb is also do well: remember to change mean)
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)  # back original size
        return img_id[1], gt

'''
YOLOv3
'''
class ListDataset(Dataset):
    def __init__(self, root,image_set=[('2012', 'trainval')], img_size=416, target_transform=None):
        self.root = root
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_set:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channel = img.shape

        dim_diff = np.abs(height - width)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if height <= width else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        filled_labels[range(len(target))[:self.max_objects]] = target[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return input_img, filled_labels

    def __len__(self):
        return len(self.ids)

class VOCSegDataSet(Dataset):
    def __init__(self, root='/input/VOC2012', phase="train",is_transform=False, img_size=512, augmentations=None, img_norm=True):
        self.phase = phase
        self.is_transform = is_transform
        self.img_size = img_size
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.root = root
        self.infoFile = [os.path.join(self.root, "ImageSets", "Segmentation", "%s.txt" % phase),
                         os.path.join(self.root, "ImageSets", "Segmentation", "benchmark_%s.txt" % phase)]

        lines = tuple(open(self.infoFile[0], 'r'))
        lines1 = tuple(open(self.infoFile[1], 'r'))
        self.file_list = [id.rstrip() for id in lines] + [id.rstrip() for id in lines1]
        self.mean = np.array([104.00699, 116.66877, 122.67892])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_index = self.file_list[index]
        im_path = os.path.join(self.root,  'JPEGImages',  file_index + '.jpg')
        lbl_path = os.path.join(self.root, 'SegmentationImages', file_index + '.png')
        im = imread(im_path)
        im = np.array(im, dtype=np.uint8)
        lbl = imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int8)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        img = imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        lbl = imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest',
                         mode='F')
        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

