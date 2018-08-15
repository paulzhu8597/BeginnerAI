import torch
from torch.nn import Linear, Sequential, ReLU, MSELoss
import numpy as np
from torch.autograd import Variable
from torch.optim import SGD, RMSprop
from lib.ProgressBar import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

STEPS = 30000
DECAY_STEP = 100
np.random.seed(0)
np.set_printoptions(linewidth=1000)

x = np.array([1.40015721,1.76405235,2.97873798,4.02272212,5.2408932,5.86755799,6.84864279,6.95008842,7.89678115])
y = np.array([-6.22959012,-6.80028513,-4.58779845,-2.1475575,3.62506375,8.40186804,16.84301125,18.99745441,27.56686965])

x_data = torch.FloatTensor(x)
y_data = torch.FloatTensor(y)

x_data = torch.unsqueeze(x_data, dim=1)
y_data = torch.unsqueeze(y_data, dim=1)

Net = Sequential(
    Linear(in_features=1, out_features=10),
    ReLU(inplace=True),
    Linear(in_features=10, out_features=1)
)

optimizer = RMSprop(Net.parameters(), lr=0.0005)
loss_func = MSELoss()

x_data, y_data = Variable(x_data), Variable(y_data)
bar = ProgressBar(1, STEPS, "train_loss:%.9f")

predict = []
myloss = []

for step in range(STEPS):
    prediction = Net(x_data)
    loss = loss_func(prediction, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    bar.show(1, loss.item())
    if (step + 1) % DECAY_STEP == 0:
        predict.append(prediction.data.numpy())
        myloss.append(loss.item())

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-', animated=False)
plt.scatter(x_data, y_data)
plt.grid(True)
time_template = 'step = %d, train loss=%.9f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
plt.title('Pytorch', fontsize=18)
def init():
    return ln,

def update(i):
    newx = x_data
    newy = predict[i]
    ln.set_data(newx, newy)
    time_text.set_text(time_template % (i * DECAY_STEP, myloss[i]))
    return ln,

ani = animation.FuncAnimation(fig, update, frames=range(int(STEPS / DECAY_STEP)),
                              init_func=init, interval=50)
ani.save("../results/02_13_07.gif", writer='imagemagick', fps=100)
# plt.show()
