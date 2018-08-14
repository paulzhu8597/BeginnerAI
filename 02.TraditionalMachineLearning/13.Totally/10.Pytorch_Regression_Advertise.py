import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import torch
from torch.nn import Linear, Sequential, ReLU, MSELoss, Sigmoid
import numpy as np
from torch.autograd import Variable
from torch.optim import SGD, Adam, RMSprop
from lib.utils.ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

STEPS = 30000
DECAY_STEP = 100

# pandas读入
dataFile = 'data/Advertising.csv'
data = pd.read_csv(dataFile)
x = data[['TV', 'Radio']]
y = data['Sales']

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)


order = y_train.argsort(axis=0)
y_train = y_train.values[order]
y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
x_train = x_train.values[order, :]

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)

Net = Sequential(
    # BatchNorm1d(num_features=2),
    Linear(in_features=2, out_features=10),

    ReLU(inplace=True),
    Linear(in_features=10, out_features=1),
)

optimizer = RMSprop(Net.parameters(), lr=0.001)
loss_func = MSELoss()

x_data, y_data = Variable(x_train), Variable(y_train)
bar = ProgressBar(1, STEPS, "train_loss:%.9f")

predict = []
myloss = []

for step in range(STEPS):
    prediction = Net(x_data)
    loss = loss_func(prediction, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bar.show(loss.item())
    if (step + 1) % DECAY_STEP == 0:
        predict.append(prediction.data.numpy())
        myloss.append(loss.item())
#
fig, ax = plt.subplots()
t = np.arange(len(x_data))
ln, = ax.plot([], [], 'r-', animated=False)
plt.scatter(t, y_data)
plt.title('Pytorch', fontsize=18)
time_template = 'step = %d, train loss=%.9f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
plt.grid(True)
def init():

    return ln,

def update(i):
    newx = t
    newy = predict[i]
    ln.set_data(newx, newy)
    time_text.set_text(time_template % (i * DECAY_STEP, myloss[i]))
    return ln,

ani = animation.FuncAnimation(fig, update, frames=range(int(STEPS / DECAY_STEP)),
                              init_func=init, interval=50)
# ani.save("Pytorch_Advertisement.gif", writer='imagemagick', fps=100)
plt.show()

order = y_test.argsort(axis=0)
y_test = y_test.values[order]
y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
x_test = x_test.values[order, :]

x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

test_predict = Net(Variable(x_test))

mse = np.average((test_predict.data.numpy() - np.array(y_test)) ** 2)
print("MSE is :%.3f" % mse)

plt.figure(facecolor='w')
t = np.arange(len(x_test))
plt.scatter(t, y_test)
plt.plot(t, test_predict.data.numpy(), 'g-', linewidth=2, label=u'预测数据, MSE:%.3f' % mse)
plt.legend(loc='upper left')
plt.title('Pytorch', fontsize=18)
plt.grid(b=True)
plt.show()
