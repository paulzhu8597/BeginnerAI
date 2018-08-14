import os
import numpy as np
import pandas as pd

from keras import layers as klayers
from keras import models as kmodels
from keras import utils as kutils
from keras import optimizers as kopts
from tqdm import tqdm
from imageio import imread
from skimage.transform import resize as imresize
from sklearn.model_selection import train_test_split
from keras.preprocessing import image as kpimage
from keras import callbacks as kcallbacks
from libs.common import ProgressBarCallback

GPU_NUMS = 1
DATA_PATH = os.path.join("../", "input")
BATCH_SIZE = 16
EPOCHS = 30
CLASS = {
    "Black-grass" : 0,
    "Charlock" : 1,
    "Cleavers" : 2,
    "Common Chickweed" : 3,
    "Common wheat" : 4,
    "Fat Hen" : 5,
    "Loose Silky-bent" : 6,
    "Maize" : 7,
    "Scentless Mayweed" : 8,
    "Shepherds Purse" : 9,
    "Small-flowered Cranesbill" : 10,
    "Sugar beet" : 11
}

INV_CLASS = {
    0 : "Black-grass",
    1 : "Charlock",
    2 : "Cleavers",
    3 : "Common Chickweed",
    4 : "Common wheat",
    5 : "Fat Hen",
    6 : "Loose Silky-bent",
    7 : "Maize",
    8 : "Scentless Mayweed",
    9 : "Shepherds Purse",
    10 : "Small-flowered Cranesbill",
    11 : "Sugar beet"
}

def loadFile(folder):
    fileList = []
    for root, dirs, files in os.walk(folder):
        if dirs == []:
            for f in files:
                filepath = os.path.join(root, f)
                fileList.append(filepath)

    file_dict = {
        "image" : [],
        "label" : [],
        "class" : []
    }

    text = ''
    if 'train' in fileList[0]:
        text = 'Start fill train_dict'
    elif 'test' in fileList[0]:
        text = 'Start fill test_dict'

    for p in tqdm(fileList, ascii=True, ncols=85, desc=text):
        image = imread(p)
        image = imresize(image, (51,51,3))
        file_dict['image'].append(image)
        file_dict['label'].append(str(str(p.split('/')[-1])))

        if 'train' in p:
            file_dict['class'].append(str(p.split('/')[-2]))
    return file_dict

def getModel():
    input_img = klayers.Input(shape=(51,51,3))

    network = klayers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1))(input_img)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1))(network)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(network)

    network = klayers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1))(network)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1))(network)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(network)

    network = klayers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))(network)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))(network)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))(network)
    network = klayers.BatchNormalization(axis=3)(network)
    network = klayers.LeakyReLU(alpha=0.1)(network)

    network = klayers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(network)

    network = klayers.Flatten()(network)
    network = klayers.Dropout(rate=0.)(network)
    network = klayers.Dense(units=128)(network)
    network = klayers.BatchNormalization(axis=-1)(network)
    network = klayers.Activation(activation="tanh")(network)

    network = klayers.Dropout(rate=0.)(network)
    network = klayers.Dense(units=12)(network)
    network = klayers.BatchNormalization(axis=-1)(network)
    network = klayers.Activation(activation="softmax")(network)

    model = kmodels.Model(inputs=input_img, outputs=network)

    if GPU_NUMS >= 2:
        model = kutils.multi_gpu_model(model, gpus=GPU_NUMS)

    model.compile(loss='categorical_crossentropy', optimizer=kopts.SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    return model

def train(model, img, target):
    x_train, x_valid, y_train, y_valid = train_test_split(
        img,
        target,
        shuffle=True,
        train_size=0.8,
        random_state=11
    )
    gen = kpimage.ImageDataGenerator(
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )
    lr_reduce = kcallbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=6, verbose=1)
    proBar = ProgressBarCallback()
    model.fit_generator(gen.flow(x_train, y_train,batch_size=BATCH_SIZE),
                         steps_per_epoch=10*len(x_train)/BATCH_SIZE,
                         epochs=EPOCHS,
                         verbose=0,
                         shuffle=True,
                         validation_data=(x_valid, y_valid),
                         callbacks=[lr_reduce, proBar])

def test(img, label):
    gmodel = getModel()
    gmodel.load_weights(filepath='../input/plant-weight/model_weight_SGD.hdf5')
    prob = gmodel.predict(img, verbose=1)
    pred = prob.argmax(axis=-1)
    sub = pd.DataFrame({"file": label,
                        "species": [INV_CLASS[p] for p in pred]})
    sub.to_csv("sub.csv", index=False, header=True)

if __name__ == '__main__':
    TRAIN_FOLDER_PATH = os.path.join(DATA_PATH, "train")
    train_dict = loadFile(TRAIN_FOLDER_PATH)
    X_train = np.array(train_dict['image'])
    y_train = kutils.to_categorical(np.array([CLASS[l] for l in train_dict['class']]))
    model = getModel()
    train(model, X_train, y_train)

    TEST_FOLDER_PATH = os.path.join(DATA_PATH, "test")
    test_dict = loadFile(TEST_FOLDER_PATH)
    X_test = np.array(test_dict['image'])
    label = test_dict['label']

    test(X_test, label)