import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libs.common import ProgressBarCallback
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import utils as KUtils
from keras import applications as KApps
from keras import layers as KLayers

seed = 2
np.random.seed(seed)

# Load the data
train = pd.read_csv("../input/train.csv")
print(train.shape)

y = train["label"]
X = train.drop("label", axis = 1)
print(y.value_counts().to_dict())
y = to_categorical(y, num_classes = 10)
del train

X = X / 255.0
X = X.values.reshape(-1,28,28,1)

train_index, valid_index = ShuffleSplit(n_splits=1, train_size=0.9, test_size=None, random_state=seed).split(X).__next__()
train_x = X[train_index]
train_y = y[train_index]
valid_x = X[valid_index]
valid_y = y[valid_index]
print(train_x.shape, valid_x.shape)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Valid', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same',  activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

GPU_NUM = 2
if GPU_NUM >= 2:
    model = KUtils.multi_gpu_model(model, gpus=GPU_NUM)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics=["accuracy"])
annealer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 30
batch_size = 86

datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                             samplewise_std_normalization=False, zca_whitening=False, rotation_range=10, zoom_range = 0.1, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=False, vertical_flip=False)#, preprocessing_function=random_add_or_erase_spot)
proBar = ProgressBarCallback()
model.fit_generator(datagen.flow(train_x,train_y,batch_size=batch_size),
                    epochs=epochs, validation_data=(valid_x[:600,:],valid_y[:600,:]),
                    verbose = 0, steps_per_epoch=train_x.shape[0]//batch_size,
                    callbacks=[annealer, proBar])


tr_ps =  model.predict(train_x)
tr_p = np.max(tr_ps, axis=1)
tr_pl = np.argmax(tr_ps, axis=1)
vl_ps = model.predict(valid_x)
vl_p = np.max(vl_ps, axis=1)
vl_pl = np.argmax(vl_ps, axis=1)

print('Base model scores:')
valid_loss, valid_acc = model.evaluate(valid_x, valid_y, verbose=0)
print("model valid loss: {0:.4f}, valid accuracy: {1:.4f}".format(valid_loss, valid_acc))

valid_p = np.argmax(model.predict(valid_x), axis=1)
target = np.argmax(valid_y, axis=1)
cm = confusion_matrix(target, valid_p)
print(cm)


test = pd.read_csv("../input/test.csv")
print(test.shape)
test = test / 255.0
test = test.values.reshape(-1,28,28,1)
# p = predict_label(test)
p = np.argmax(model.predict(test), axis = 1)

submission = pd.DataFrame(pd.Series(range(1, p.shape[0]+1), name='ImageId'))
submission['Label'] = p
submission.to_csv("cnn.csv", index=False)