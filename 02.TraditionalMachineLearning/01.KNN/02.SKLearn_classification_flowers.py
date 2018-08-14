'''
SKLearn-分类
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl

# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

data = pd.read_csv("../../data/iris.data", header=None)
x = data[np.arange(4)]
y = pd.Categorical(data[4]).codes
x = x.iloc[:, :2]# 为了可视化，仅使用前两列特征
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

nFolds = 4
random_state = 1234
metrics = ['minkowski','euclidean','manhattan']
weights = ['uniform','distance'] #10.0**np.arange(-5,4)
algorithm=['ball_tree','kd_tree','brute', 'auto']
numNeighbors = np.arange(5,10)
param_grid = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
cv = StratifiedKFold(nFolds)
cv.get_n_splits(x_train, y_train)
classifier = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv)

classifier.fit(x_train, y_train)
y_test_hat = classifier.predict(x_test)
print(classifier.best_params_)
N, M = 50,50 # 横纵各采样多少个值
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

y_show_hat = classifier.predict(x_show)  # 预测值
y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x_test[0], x_test[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=cm_dark, marker='*')  # 测试数据
plt.xlabel(iris_feature[0], fontsize=15)
plt.ylabel(iris_feature[1], fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)

plt.title(u'鸢尾花数据的K近邻分类\n测试集准确度: %.2f%%' % (100 * classifier.score(x_test, y_test)), fontsize=17)
plt.savefig("../results/02_01_02.png")
plt.show()