import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.animation as animation
from lib.utils.ProgressBar import ProgressBar
import pandas as pd
from sklearn.model_selection import train_test_split

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

x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

Weights_L1 = tf.Variable(tf.random_normal([2, 10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.relu(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
prediction = tf.matmul(L1, Weights_L2) + biases_L2
# prediction = tf.nn.tanh(Wx_plus_b_L2)

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

predict = []
myloss = []
bar = ProgressBar(1, STEPS, "train_loss:%.9f")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(STEPS):
    _, train_loss,prediction_value = sess.run([train_step, loss, prediction], feed_dict={x:x_train, y:y_train})

    bar.show(train_loss)
    if (step + 1) % DECAY_STEP == 0:
        predict.append(prediction_value)
        myloss.append(train_loss)

fig, ax = plt.subplots()
t = np.arange(len(x_train))
ln, = ax.plot([], [], 'r-', animated=False)
plt.scatter(t, y_train)
time_template = 'step = %d, train loss=%.9f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
plt.title('Tensorflow', fontsize=18)
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
# ani.save("Tensorflow_Advertisement.gif", writer='imagemagick', fps=100)
plt.show()

order = y_test.argsort(axis=0)
y_test = y_test.values[order]
y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
x_test = x_test.values[order, :]

test_predict = sess.run(prediction,feed_dict={x:x_test})
mse = np.average((test_predict - np.array(y_test)) ** 2)

plt.figure(facecolor='w')
t = np.arange(len(x_test))
plt.scatter(t, y_test)
plt.plot(t, test_predict, 'g-', linewidth=2, label=u'预测数据, MSE:%.3f' % mse)
plt.legend(loc='upper left')
plt.title('Tensorflow', fontsize=18)
plt.grid(b=True)
plt.show()