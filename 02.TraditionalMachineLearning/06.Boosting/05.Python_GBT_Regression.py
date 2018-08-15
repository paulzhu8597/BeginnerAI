import numpy as np
import lib.tml.decisiontree as dtree
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

def calculate_variance(X):
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance
class RegressionTree(dtree.DecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # 使用方差缩减
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class GradientBoosting(object):
    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        self.loss = SquareLoss()

        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth)
            self.trees.append(tree)


    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # Update y prediction
            y_pred -= np.multiply(self.learning_rate, update)


    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples_split=min_samples_split,
                                                        min_impurity=min_var_red,
                                                        max_depth=max_depth,
                                                        regression=True)

data = pd.read_csv('../../data/MyData.txt', sep="\t")

time = np.atleast_2d(data["time"].as_matrix()).T
temp = np.atleast_2d(data["temp"].as_matrix()).T

X = time.reshape((-1, 1))               # Time. Fraction of the year [0, 1]
X = np.insert(X, 0, values=1, axis=1)   # Insert bias term
y = temp[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred_line = model.predict(X)
mse = mean_squared_error(y_test, y_pred)
plt.figure(figsize=(10,10), facecolor='w')

m1 = plt.scatter(366 * X_train[:, 1],y_train, c='r',s=10, label=u'训练集')
m2 = plt.scatter(366 * X_test[:, 1], y_test,c='g',s=10, label=u'测试集')
m3 = plt.scatter(366 * X_test[:, 1], y_pred,c='b',s=10, label=u'预测集')
plt.legend(loc='best')
plt.savefig("../results/02_06_05.png")
plt.show()

