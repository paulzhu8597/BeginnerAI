import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

style = ''
np.random.seed(0)
mu1_fact = (0, 0, 0)
cov1_fact = np.diag((1, 2, 3))
data1 = np.random.multivariate_normal(mu1_fact, cov1_fact, 400)
mu2_fact = (2, 2, 1)
cov2_fact = np.array(((1, 1, 3), (1, 2, 1), (0, 0, 1)))
data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 100)
data = np.vstack((data1, data2))
y = np.array([True] * 400 + [False] * 100)

fig = plt.figure(figsize=(18, 7), facecolor='w')
ax = fig.add_subplot(131, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(u'原始数据', fontsize=18)

g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
g.fit(data)
sklearn_mu1, sklearn_mu2 = g.means_
sklearn_sigma1, sklearn_sigma2 = g.covariances_
norm1 = multivariate_normal(sklearn_mu1, sklearn_sigma1)
norm2 = multivariate_normal(sklearn_mu2, sklearn_sigma2)
tau1 = norm1.pdf(data)
tau2 = norm2.pdf(data)

ax = fig.add_subplot(132, projection='3d')
order = pairwise_distances_argmin([mu1_fact, mu2_fact], [sklearn_mu1, sklearn_mu2], metric='euclidean')
if order[0] == 0:
    c1 = tau1 > tau2
else:
    c1 = tau1 < tau2
c2 = ~c1
acc = np.mean(y == c1)
print(u'SkLearn准确率：%.2f%%' % (100*acc))
ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(u'SKLearn-EM算法分类', fontsize=18)


# 预测分类
num_iter = 100
n, d = data.shape

python_mu1 = data.min(axis=0)
python_mu2 = data.max(axis=0)
python_sigma1 = np.identity(d)
python_sigma2 = np.identity(d)
pi = 0.5
# EM
for i in range(num_iter):
    # E Step
    norm1 = multivariate_normal(python_mu1, python_sigma1)
    norm2 = multivariate_normal(python_mu2, python_sigma2)
    tau1 = pi * norm1.pdf(data)
    tau2 = (1 - pi) * norm2.pdf(data)
    gamma = tau1 / (tau1 + tau2)

    # M Step
    python_mu1 = np.dot(gamma, data) / np.sum(gamma)
    python_mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
    python_sigma1 = np.dot(gamma * (data - python_mu1).T, data - python_mu1) / np.sum(gamma)
    python_sigma2 = np.dot((1 - gamma) * (data - python_mu2).T, data - python_mu2) / np.sum(1 - gamma)
    pi = np.sum(gamma) / n

norm1 = multivariate_normal(python_mu1, python_sigma1)
norm2 = multivariate_normal(python_mu2, python_sigma2)
tau1 = norm1.pdf(data)
tau2 = norm2.pdf(data)

ax = fig.add_subplot(133, projection='3d')
order = pairwise_distances_argmin([mu1_fact, mu2_fact], [python_mu1, python_mu2], metric='euclidean')

if order[0] == 0:
    c1 = tau1 > tau2
else:
    c1 = tau1 < tau2
c2 = ~c1
acc = np.mean(y == c1)
print(u'Python准确率：%.2f%%' % (100*acc))
ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(u'Python-EM算法分类', fontsize=18)

plt.suptitle(u'EM算法的实现', fontsize=21)
plt.subplots_adjust(top=0.90)
plt.tight_layout()
plt.savefig("../results/02_10_01.png")
plt.show()