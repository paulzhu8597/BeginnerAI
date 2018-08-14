import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn import datasets

class NaiveBayes():
    """The Gaussian Naive Bayes classifier. """
    def fit(self, X, y):

        """
        X [shape,features]
        y [shape,label]
        """

        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        # 计算每一个类别的每一个特征的方差和均值
        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            # 计算每一个特征
            for j in range(X.shape[1]):
                col = X_where_c[:, j] #列
                parameters = {"mean": col.mean(), "var": col.var()} #求方差 与 均值
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        """ 计算高斯概率密度 输入均值 和 方差"""
        eps = 1e-4 # Added in denominator to prevent division by zero
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent
    def _calculate_prior(self, c):
        """ 计算先验概率 """
        X_where_c = self.X[np.where(self.y == c)]
        n_class_instances = X_where_c.shape[0]
        n_total_instances = self.X.shape[0]
        return n_class_instances / n_total_instances

    def _classify(self, sample):
        posteriors = []
        for i, c in enumerate(self.classes):
            # 计算每一个类别的先验概率 p(y=c)=?
            posterior = self._calculate_prior(c)

            for j, params in enumerate(self.parameters[i]):
                # 提取每一个类别下的特征值的方差 以及 均值
                sample_feature = sample[j]
                # 计算高斯密度
                likelihood = self._calculate_likelihood(params["mean"], params["var"], sample_feature)
                posterior *= likelihood
            posteriors.append(posterior)
        # 求最大概率对应的类别
        index_of_max = np.argmax(posteriors)
        return self.classes[index_of_max]
    def predict(self, X):
        y_pred = []
        for sample in X:
            y = self._classify(sample)
            y_pred.append(y)
        return y_pred

data = datasets.load_digits()
X = normalize(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = NaiveBayes()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred,y_test))