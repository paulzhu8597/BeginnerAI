import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

dataLoad = pd.read_csv("../../data/watermelon30a.txt", header=None)
x,y = dataLoad.values[:, :-1], dataLoad[2].tolist()
y = pd.Categorical(y).codes
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.45, random_state=123,
                                                    stratify=y)

print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(train_y) / float(len(train_y)) * 100.0)
print('Test:', np.bincount(test_y) / float(len(test_y)) * 100.0)

metrics = ['minkowski','euclidean','manhattan']
weights = ['uniform','distance'] #10.0**np.arange(-5,4)
algorithm=['ball_tree','kd_tree','brute', 'auto']
numNeighbors = np.arange(1,4)
param_grid = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
classifier =GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)

print(classifier.best_params_)

accuracy = np.sum(pred_y == test_y) / float(len(test_y))
print(accuracy)

print("Samples correctly classified:")
correct_idx = np.where(pred_y == test_y)[0]
print(correct_idx)

print("Samples incorrectly classified:")
incorrect_idx = np.where(pred_y != test_y)[0]
print(incorrect_idx)
colors = ["darkblue","darkgreen"]
plt.figure(facecolor='w')
for n, color in enumerate(colors):
    idx = np.where(test_y == n)[0]
    plt.scatter(test_x[idx, 0],test_x[idx, 1],c=color, s =40, label="Class %s" % n)

plt.scatter(test_x[incorrect_idx, 0],test_x[incorrect_idx,1],c='darkred', s =40)
plt.xlabel("sepal width [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.title("Watermelon Classification results")
plt.savefig("../results/02_01_04.png")
plt.show()