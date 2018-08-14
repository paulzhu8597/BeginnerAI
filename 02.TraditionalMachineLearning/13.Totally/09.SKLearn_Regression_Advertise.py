import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn import svm
import pandas as pd
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
data = pd.read_csv('data/Advertising.csv')
x = data[['TV', 'Radio']]
y = data['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

modelList = [
    [KNeighborsRegressor(), 'K近邻'],
    [Pipeline([('poly', PolynomialFeatures()),
               ('linear', LinearRegression(fit_intercept=False))]), '线性回归'],
    [Pipeline([('poly', PolynomialFeatures()),
               ('linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]), '岭回归'],
    [Pipeline([('poly', PolynomialFeatures(degree=6)),
               ('linear', LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]), 'LASSO回归'],
    [Pipeline([('poly', PolynomialFeatures(degree=6)),
               ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 50),
                                       l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                       fit_intercept=False))]), '弹性网'],
    [DecisionTreeRegressor(criterion='mse'), '决策树'],
    [RandomForestRegressor(n_estimators=10), '随机森林'],
    [GradientBoostingRegressor(n_estimators=100), 'GBRT'],
    [xgb.XGBRegressor(n_estimators=200, max_depth=4), 'XGBoosting'],
    [AdaBoostRegressor(n_estimators=100), 'AdaBoost'],
    [svm.SVR(kernel='linear'), 'SVR']
]

plt.figure(figsize=(14, 14 ), facecolor='w')
order = y_test.argsort(axis=0)
x_test = x_test.values[order, :]
y_test = y_test.values[order]
x_test = pd.DataFrame(x_test)
x_test.columns = ['TV', 'Radio']
x_hat = np.arange(len(x_test))
for i, clf in enumerate(modelList):
    model = clf[0]
    title = clf[1]
    model.fit(x_train, y_train)

    y_hat = model.predict(x_test)

    plt.subplot(4, 3, i+1)
    plt.scatter(x_hat, y_test)

    r2 = model.score(x_test, y_test)

    label = u'$R^2$=%.3f' % (r2)
    plt.plot(x_hat, y_hat, 'r-',linewidth=2, label=label)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(title, fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.suptitle('广告销量预测',fontsize=18)
plt.tight_layout(1.5)
plt.subplots_adjust(top=0.92)
# plt.imsave("~/Desktop/广告销量预测_各种算法比较.png")
plt.show()