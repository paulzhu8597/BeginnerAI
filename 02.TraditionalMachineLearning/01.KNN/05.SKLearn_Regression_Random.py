import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(0)
np.set_printoptions(linewidth=1000)
N = 9
x = np.array([1.40015721,1.76405235,2.97873798,4.02272212,5.2408932,5.86755799,6.84864279,6.95008842,7.89678115])
y = np.array([-6.22959012,-6.80028513,-4.58779845,-2.1475575,3.62506375,8.40186804,16.84301125,18.99745441,27.56686965])

x.shape = -1, 1
y.shape = -1, 1

metrics = ['minkowski','euclidean','manhattan']
weights = ['uniform','distance'] #10.0**np.arange(-5,4)
algorithm=['ball_tree','kd_tree','brute', 'auto']
numNeighbors = np.arange(4,6)
param_grid = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
model = GridSearchCV(KNeighborsRegressor(),param_grid=param_grid)
model.fit(x, y)
print(model.best_params_)
x_hat = np.linspace(x.min(), x.max(), num=10)
x_hat.shape = -1, 1
y_hat = model.predict(x_hat)
plt.figure(figsize=(10, 10), facecolor='w')
plt.plot(x, y, 'ro', ms=10, zorder=N)
s = model.score(x, y)
label = u'$R^2$=%.3f' % (s)
plt.plot(x_hat, y_hat, color='r', lw=2, label=label, alpha=0.75)
plt.legend(loc='upper left')
plt.grid(True)
plt.title('KNN回归', fontsize=18)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.savefig("../results/02_01_05.png")
plt.show()