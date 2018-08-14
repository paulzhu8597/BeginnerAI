'''
Python实现KNN算法
'''
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def computeMinkowskiDistance(vector1, vector2, q ):
    distance = 0.
    n = len(vector1)
    for i in range(n):
        distance += pow(abs(float(vector1[i]) - float(vector2[i])), q)
    return round(pow(distance, 1.0 / q), 5)

def computeManhattanDistance(vector1, vector2):
    return computeMinkowskiDistance(vector1, vector2, 1)

def computeEuDistance(vector1, vector2):
    return computeMinkowskiDistance(vector1, vector2, 2)

class KNN(object):
    def __init__(self,k=5):
        self.k = 5
    def _vote(self,neighbours):
        counts = np.bincount(neighbours[:, 1].astype('int'))
        return counts.argmax()
    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # 对每一个test进行循环
        for i,test in enumerate(X_test):
            neighbours = np.empty((X_train.shape[0],2))
            # 对每一个train进行计算
            for j, train in enumerate(X_train):
                dis = computeEuDistance(train,test)
                label = y_train[j]
                neighbours[j] = [dis,label]
            k_nearest_neighbors = neighbours[neighbours[:,0].argsort()][:self.k]
            label = self._vote(k_nearest_neighbors)
            y_pred[i] = label
        return y_pred
data = datasets.load_iris()
X = normalize(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
clf = KNN(k=5)
y_pred = clf.predict(X_test, X_train, y_train)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)