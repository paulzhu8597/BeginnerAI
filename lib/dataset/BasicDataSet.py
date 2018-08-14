# coding=utf-8
import numpy as np
import pickle
import random
import os

class BasicDataSet(object):
    def __init__(self, root, train_ratio=1):
        self._root_path = root
        self.ratio=train_ratio

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def read(self, onehot=False, channel_first=True):
        (x_train, targets_train), (x_test, targets_test) = self._readData(channel_first)

        x_total = np.concatenate((x_train, x_test))
        y_total = np.concatenate((targets_train, targets_test))

        index_list = list(range(0, x_total.shape[0]))
        random.shuffle(index_list)

        train_record_count = int(len(index_list) * self.ratio)

        index_train = index_list[0:train_record_count]
        index_test  = index_list[train_record_count:len(index_list)]

        x_train = x_total[index_train]
        x_test = x_total[index_test]
        targets_train = y_total[index_train]
        targets_test = y_total[index_test]

        self.TRAIN_RECORDS = x_train.shape[0]
        self.TEST_RECORDS = x_test.shape[0]

        if onehot:
            y_train = np.zeros((targets_train.shape[0], 10), dtype = np.uint8)
            y_test = np.zeros((targets_test.shape[0], 10), dtype = np.uint8)
            y_train[np.arange(targets_train.shape[0]), targets_train] = 1
            y_test[np.arange(targets_test.shape[0]), targets_test] = 1

            return (x_train, y_train), (x_test, y_test)
        else:
            return (x_train, np.reshape(targets_train, newshape=(targets_train.shape[0], 1))), (x_test, np.reshape(targets_test, newshape=(targets_test.shape[0], 1)))


    def _readData(self, channel_first):
        pass

class Cifar10DataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_CHANNEL = 3
    IMAGE_SIZE = 32
    TRAIN_RECORDS = 50000
    TEST_RECORDS = 10000
    LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    def __init__(self, root="/input/cifar10p/", train_ratio=0.9, special_label=None):
        super(Cifar10DataSet, self).__init__(root=root, train_ratio=train_ratio)
        self.special_label = special_label

    def _readData(self, channel_first):
        data_train = []
        targets_train = []
        for i in range(5):
            raw = self.unpickle(self._root_path + 'data_batch_' + str(i + 1))
            data_train += [raw[b'data']]
            targets_train += raw[b'labels']

        data_train = np.concatenate(data_train)
        targets_train = np.array(targets_train)

        data_test = self.unpickle(self._root_path + 'test_batch')
        targets_test = np.array(data_test[b'labels'])
        data_test = np.array(data_test[b'data'])

        x_train = np.reshape(data_train, (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))
        x_test  = np.reshape(data_test,  (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))

        if channel_first == False:
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        if self.special_label is not None:
            x_train = x_train[np.where(targets_train==self.special_label)[0]]
            targets_train = targets_train[targets_train[:]==self.special_label]
            x_test = x_test[np.where(targets_test==self.special_label)[0]]
            targets_test = targets_test[targets_test[:]==self.special_label]

        return (x_train, targets_train), (x_test, targets_test)

class MnistDataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_SIZE = 28
    IMAGE_CHANNEL = 1

    def __init__(self, root="/input/mnist.npz", radio=1):
        super(MnistDataSet, self).__init__(root=root, train_ratio=radio)

    def _readData(self, channel_first):
        f = np.load(self._root_path)
        x_train, targets_train = f['x_train'], f['y_train']
        x_test, targets_test = f['x_test'], f['y_test']
        f.close()

        return (x_train, targets_train), (x_test, targets_test)

class STLDataSet(BasicDataSet):
    NUM_OUTPUTS = 10
    IMAGE_CHANNEL = 3
    IMAGE_SIZE = 96
    TRAIN_RECORDS = 5000
    TEST_RECORDS = 8000
    LABELS = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

    def __init__(self, root="/input/STLB/"):
        super(STLDataSet, self).__init__(root=root, train_ratio=0.95)
        self.train_image_file = os.path.join(self._root_path, "train_X.bin")
        self.train_label_file = os.path.join(self._root_path, "train_y.bin")
        self.test_image_file = os.path.join(self._root_path, "test_X.bin")
        self.test_label_file = os.path.join(self._root_path, "test_y.bin")

    def _readData(self, channel_first):

        with open(self.train_image_file, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            x_train = np.reshape(everything, (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))

        with open(self.train_label_file, 'rb') as f:
            targets_train = np.fromfile(f, dtype=np.uint8)

        with open(self.test_image_file, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            x_test = np.reshape(everything, (-1, self.IMAGE_CHANNEL, self.IMAGE_SIZE, self.IMAGE_SIZE))

        with open(self.test_label_file, 'rb') as f:
            targets_test = np.fromfile(f, dtype=np.uint8)

        targets_train = targets_train-1
        targets_test = targets_test - 1
        if channel_first == False:
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, targets_train), (x_test, targets_test)


