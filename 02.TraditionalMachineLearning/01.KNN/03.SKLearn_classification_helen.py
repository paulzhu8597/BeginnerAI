import pandas as pd
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

dataLoad = pd.read_csv("../../data/helen_date.txt", header=None)
x = dataLoad[np.arange(4)]
y = dataLoad[3]
Nor_x = x.iloc[:, :2]
metrics = ['minkowski','euclidean','manhattan']
weights = ['uniform','distance'] #10.0**np.arange(-5,4)
algorithm=['ball_tree','kd_tree','brute', 'auto']
numNeighbors = np.arange(5,10)
param_grid = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)

model = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
model.fit(Nor_x, y.tolist())
N, M = 50,50 # 横纵各采样多少个值
x1_min, x2_min = Nor_x.min()
x1_max, x2_max = Nor_x.max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

y_show_hat = model.predict(x_show)  # 预测值
y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同

colors ={"1":"orange", "2" : "black", "3" : "red"}

plt.figure(figsize=(13,13), facecolor='w')
for index, config in enumerate([[0,1],[0,2],[1,2], [0,1]]):
    plt.subplot(2,2, index + 1)
    data_x_index = config[0]
    data_y_index = config[1]
    if (index == 3):
        data_x = x.values[:, data_x_index].tolist()
        data_y = x.values[:, data_y_index].tolist()
        plt.pcolormesh(x1, x2, y_show_hat, cmap=mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF']))  # 预测值的显示
    else:
        data_x = x.values[:, data_x_index].tolist()
        data_y = x.values[:, data_y_index].tolist()
    plt.scatter(data_x, data_y, c=y.ravel(), cmap=mpl.colors.ListedColormap(['orange', 'black', 'red']))
    plt.grid(True)

plt.title(u'Helen约会数据的K近邻分类')
plt.savefig("../results/02_01_03.png")
plt.show()