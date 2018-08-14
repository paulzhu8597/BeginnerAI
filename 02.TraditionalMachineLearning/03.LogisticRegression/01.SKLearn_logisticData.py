import numpy as np
import matplotlib.pyplot as plt
from lib.ProgressBar import ProgressBar

def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                        #创建标签列表
    fr = open('../../data/logisticData.txt')                                            #打开文件
    for line in fr.readlines():                                            #逐行读取
        lineArr = line.strip().split()                                    #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])        #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                            #关闭文件
    return dataMat, labelMat

X, Y = loadDataSet()
# data = load_iris()
# X = data.data[data.target != 0]
# y = data.target[data.target != 0]
# y[y == 1] = 0
# y[y == 2] = 1
# Y = y.reshape(-1,1)
X = np.asarray(X)
Y = np.asarray(Y)
Y = Y.reshape(-1,1)
bar = ProgressBar(1, 1000, "loss:%.3f")
class LogisticRegression(object):
    def __init__(self):
        self.sigmoid = lambda x:1./(1+np.exp(-x))
    def fit(self, X, y):
        self.w = np.random.randn(X.shape[1],1)
        for _ in range(10000):
            y_pred = self.sigmoid(X @ self.w)
            #self.w += 0.01 * X.T @ (y - y_pred) #梯度上升
            self.w -= 0.0001 * X.T @ (y_pred - y) # 梯度下降
            #bar.show(np.mean(0.5*(y_pred- y)**2))
    def predict(self,X):
        y_pred = np.round(self.sigmoid(X.dot(self.w)))
        return y_pred
lr = LogisticRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)
accuracy = np.sum(Y == y_pred, axis=0) / len(Y)
print('predict acc %s' % accuracy)

plt.figure(facecolor='w')
plt.scatter(X[:,1], X[:,2], c = Y[:, 0])
x = np.arange(-3.0, 3.0, 0.1)
y = (-lr.w[0] - lr.w[1] * x) / lr.w[2]
plt.plot(x, y)
plt.savefig("../results/02_03_01.png")
plt.show()