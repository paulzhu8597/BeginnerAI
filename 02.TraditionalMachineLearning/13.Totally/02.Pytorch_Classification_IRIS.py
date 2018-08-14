import numpy as np
import pandas as pd
import matplotlib as mpl
import torch

from lib.utils.ProgressBar import ProgressBar
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable

STEPS = 30000
DECAY_STEP = 100
# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

data = pd.read_csv('data/iris.data', header=None)
x_data = data[np.arange(4)]
y_data = pd.Categorical(data[4]).codes

x_data = x_data.iloc[:, :2]# 为了可视化，仅使用前两列特征
x_train = x_data.values[:]
y_train = y_data.astype(int)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x, y = Variable(x_train), Variable(y_train)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=3)     # define the network

optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

bar = ProgressBar(1, STEPS, "Loss:%.9f, Accuracy:%.3f")
predict = []
myloss = []
N, M = 50,50 # 横纵各采样多少个值
x1_min, x2_min = x_data.min()
x1_max, x2_max = x_data.max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

for step in range(STEPS):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    _, prediction = torch.max(F.softmax(out, dim=None), 1)
    pred_y = prediction.data.numpy().squeeze()
    target_y = y.data.numpy()
    accuracy = sum(pred_y == target_y)/x.shape[0]

    bar.show(loss.item(), accuracy)

    if (step + 1) % DECAY_STEP == 0:
        out = net(Variable(torch.FloatTensor(x_show)))
        _, prediction = torch.max(F.softmax(out, dim=None), 1)
        pred_y = prediction.data.numpy().squeeze()
        predict.append(pred_y)
        myloss.append(loss.item())


fig, axes = plt.subplots()
plt.xlabel(iris_feature[0], fontsize=15)
plt.ylabel(iris_feature[1], fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.suptitle("Pytorch")
time_template = 'step = %d, train loss=%.9f'

def animate(i):
    plt.title(time_template % (i * DECAY_STEP, myloss[i]))
    plt.pcolormesh(x1, x2, predict[i].reshape(x1.shape), cmap=cm_light)
    plt.scatter(x_data[0], x_data[1], c=y_data.ravel(), edgecolors='k', s=40, cmap=cm_dark)


anim = animation.FuncAnimation(fig, animate, frames=range(int(STEPS / DECAY_STEP)),
                               blit=False,interval=50)
# anim.save("Pytorch_FlowerDeLuceTwoFeatures.gif", writer='imagemagick', fps=100)
plt.show()
