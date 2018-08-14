'''
使用Python实现的回归算法
'''
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def loadDataSet():
    dataLoad = pd.read_csv("../../data/randomcurve.txt", header=None)
    x = dataLoad.iloc[:, :2].values
    y = dataLoad.values[:, 2]
    return x, y

def calculateW(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat

    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse" )
        return

    ws = xTx.I * (xMat.T * yMat)
    return ws


def regress1(x, y):
    ws = calculateW(x,y)
    xMat = np.mat(x)
    yMat = np.mat(y)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws

    return (xMat, yMat), (xCopy, yHat)

def regress2(x,y):
    m = np.shape(x)[0] # 样本数量
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(x[i],x,y,1.0)
    # 返回估计值
    xMat = np.mat(x)
    yMat = np.mat(y)
    xCopy = xMat.copy()
    xCopy.sort(0)
    srtInd = xMat[:,1].argsort(0)           #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort=xMat[srtInd][:,0,:]
    return (xMat, yMat), (xSort, yHat[srtInd])

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    # eye()返回一个对角线元素为1，其他元素为0的二维数组，创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    weights = np.mat(np.eye((m)))
    for j in range(m):
        # testPoint 的形式是 一个行向量的形式
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - xMat[j,:]
        # k控制衰减的速度
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def regress3(x,y):
    m = np.shape(x)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = np.zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(x[i],x,y,0.01)
    # 返回估计值
    xMat = np.mat(x)
    yMat = np.mat(y)
    xCopy = xMat.copy()
    xCopy.sort(0)
    srtInd = xMat[:,1].argsort(0)           #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort=xMat[srtInd][:,0,:]
    return (xMat, yMat), (xSort, yHat[srtInd])

def regress4(x,y):
    m = np.shape(x)[0]
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = np.zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(x[i],x,y,0.003)
    # 返回估计值
    xMat = np.mat(x)
    yMat = np.mat(y)
    xCopy = xMat.copy()
    xCopy.sort(0)
    srtInd = xMat[:,1].argsort(0)           #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort=xMat[srtInd][:,0,:]
    return (xMat, yMat), (xSort, yHat[srtInd])

x,y=loadDataSet()
list = [
    {"title" : "欠拟合", "data" : regress1(x,y)},
    {"title" : "局部加权k=1.0", "data" : regress2(x,y)},
    {"title" : "局部加权k=0.01", "data" : regress3(x,y)},
    {"title" : "局部加权k=0.003", "data" : regress4(x,y)}
]
plt.figure(figsize=(20,5), facecolor='w')

for index, ws in enumerate(list):
    plt.subplot(1,4, index + 1)
    title = ws["title"]
    (xMat, yMat), (xCopy, yHat) = ws["data"]

    plt.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]], c='r')

    plt.plot(xCopy[:, 1], yHat, c='b')
    plt.grid(True)
    plt.title(title)
plt.savefig("02_02_01.png")
plt.show()