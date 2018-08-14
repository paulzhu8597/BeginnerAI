import numpy as np
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from time import time

'''
200 - 96.702%, 96.700
'''

print('载入训练数据...')
t = time()
data = pd.read_csv('../input/train.csv', header=0, dtype=np.int)
print('载入完成，耗时%f秒' % (time() - t))
y = data['label'].values
x = data.values[:, 1:]
print('图片个数：%d，图片像素数目：%d' % x.shape)
images = x.reshape(-1, 28, 28)
y = y.ravel()

print('载入测试数据...')
t = time()
data_test = pd.read_csv('../input/test.csv', header=0, dtype=np.int)
data_test = data_test.values
images_test_result = data_test.reshape(-1, 28, 28)
print('载入完成，耗时%f秒' % (time() - t))

np.random.seed(0)
x, x_test, y, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
images = x.reshape(-1, 28, 28)
images_test = x_test.reshape(-1, 28, 28)
print(x.shape, x_test.shape)



model = RandomForestClassifier(100, criterion='gini', min_samples_split=2,
                               min_impurity_decrease=1e-10, bootstrap=True, oob_score=True)
params_list = {"n_estimators" : [50, 100, 200, 500, 1000],
               "max_features" : [.1, .2, .3, .5, .9, .99]}

# model = GridSearchCV(RandomForestClassifier(), param_grid = params_list, cv=2)
model = RandomForestClassifier(n_estimators=200)
print('随机森林开始训练...')
t = time()
model.fit(x, y)
# print(model.best_score_)
# print(model.best_params_)
t = time() - t
print('随机森林训练结束，耗时%d分钟%.3f秒' % (int(t/60), t - 60*int(t/60)))
# print('OOB准确率：%.3f%%' % (model.oob_score_*100))
t = time()
y_hat = model.predict(x)
t = time() - t
print('随机森林训练集准确率：%.3f%%，预测耗时：%d秒' % (accuracy_score(y, y_hat)*100, t))
t = time()
y_test_hat = model.predict(x_test)
t = time() - t
print('随机森林测试集准确率：%.3f%%，预测耗时：%d秒' % (accuracy_score(y_test, y_test_hat)*100, t))


y_test = model.predict(data_test)
ImageId = np.arange(1, data_test.shape[0]+1, step=1)

dataFrame = pd.DataFrame({"ImageId" : ImageId, "Label" : y_test})
dataFrame.to_csv("result.csv", index=False)