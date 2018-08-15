import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn import svm
from sklearn import metrics
import xgboost as xgb
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(0)
np.set_printoptions(linewidth=1000)
N = 9
# x = np.linspace(0, 8, N) + np.random.randn(N)
# x = np.sort(x)
# y = x**2 - 4*x - 3 + np.random.randn(N)
x = np.array([1.40015721,1.76405235,2.97873798,4.02272212,5.2408932,5.86755799,6.84864279,6.95008842,7.89678115])
y = np.array([-6.22959012,-6.80028513,-4.58779845,-2.1475575,3.62506375,8.40186804,16.84301125,18.99745441,27.56686965])

x.shape = -1, 1
y.shape = -1, 1

modelList = [
    [KNeighborsRegressor(), 'K近邻'],
    [Pipeline([('poly', PolynomialFeatures(degree=7)),
               ('linear', LinearRegression(fit_intercept=False))]), '线性回归'],
    [Pipeline([('poly', PolynomialFeatures(degree=5)),
               ('linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]), '岭回归'],
    [Pipeline([('poly', PolynomialFeatures(degree=6)),
               ('linear', LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]), 'LASSO回归'],
    [Pipeline([('poly', PolynomialFeatures(degree=6)),
               ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 50),
                                       l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                       fit_intercept=False))]), '弹性网'],
    [DecisionTreeRegressor(criterion='mse',max_depth=4), '决策树'],
    [RandomForestRegressor(n_estimators=10, max_depth=3), '随机森林'],
    [GradientBoostingRegressor(n_estimators=100, max_depth=1), 'GBRT'],
    [xgb.XGBRegressor(n_estimators=200, max_depth=4), 'XGBoosting'],
    [AdaBoostRegressor(n_estimators=100), 'AdaBoost'],
    [svm.SVR(kernel='rbf', gamma=0.2, C=100), 'SVR']
]

x_hat = np.linspace(x.min(), x.max(), num=10)
x_hat.shape = -1, 1
plt.figure(figsize=(12, 12), facecolor='w')

for i, clf in enumerate(modelList):
    model = clf[0]
    title = clf[1]
    model.fit(x, y.ravel())
    y_hat = model.predict(x_hat)

    plt.subplot(4, 3, i+1)
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    y_pred = model.predict(x)
    r2 = metrics.r2_score(y, y_pred)

    label = u'$R^2$=%.3f' % (r2)
    plt.plot(x_hat, y_hat, color='r', lw=2, label=label, alpha=0.75)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(title, fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.suptitle('随机曲线的拟合',fontsize=18)
plt.tight_layout(1.5)
plt.subplots_adjust(top=0.92)
plt.savefig("../results/02_13_06.png")
plt.show()