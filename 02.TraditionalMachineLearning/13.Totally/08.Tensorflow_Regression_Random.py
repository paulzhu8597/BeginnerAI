import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.animation as animation
from lib.utils.ProgressBar import ProgressBar

STEPS = 30000
DECAY_STEP = 100
N = 9
x_data = np.linspace(0, 8, N)[:, np.newaxis] + np.random.randn(N)[:, np.newaxis]
x_data = np.sort(x_data)
y_data = x_data**2 - 4*x_data - 3 + np.random.randn(N)[:, np.newaxis]

x_data = np.array([1.40015721,1.76405235,2.97873798,4.02272212,5.2408932,5.86755799,6.84864279,6.95008842,7.89678115])[:, np.newaxis]
y_data = np.array([-6.22959012,-6.80028513,-4.58779845,-2.1475575,3.62506375,8.40186804,16.84301125,18.99745441,27.56686965])[:, np.newaxis]


x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.relu(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
prediction = tf.matmul(L1, Weights_L2) + biases_L2
# prediction = tf.nn.tanh(Wx_plus_b_L2)

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.MomentumOptimizer(0.0001, momentum=0.1).minimize(loss)

predict = []
myloss = []
bar = ProgressBar(1, STEPS, "train_loss:%.9f")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(STEPS):
        _, train_loss,prediction_value = sess.run([train_step, loss, prediction], feed_dict={x:x_data, y:y_data})

        bar.show(train_loss)
        if (step + 1) % DECAY_STEP == 0:
            predict.append(prediction_value)
            myloss.append(train_loss)

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'r-', animated=False)
plt.scatter(x_data, y_data)
time_template = 'step = %d, train loss=%.9f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
plt.title('Tensorflow', fontsize=18)
plt.grid(True)
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
# ani.save("Tensorflow_RandomCurve.gif", writer='imagemagick', fps=100)
plt.show()