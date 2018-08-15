import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print('%s正确率：%.3f%%' % (tip, acc_rate))
    return acc_rate


def load_data(file_name, is_train):
    data = pd.read_csv(file_name)  # 数据文件路径

    # 性别
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 补齐船票价格缺失值
    if len(data.Fare[data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = fare[f]

    # 年龄：使用均值代替缺失值
    if is_train:
        # 年龄：使用随机森林预测年龄缺失值
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]   # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print age_exist
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
    else:
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_hat

    # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    embarked_data = pd.get_dummies(data.Embarked)
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    # 思考：这样做，其实发生了什么？
    x = np.tile(x, (5, 1))
    y = np.tile(y, (5, ))
    if is_train:
        return x, y
    return x, data['PassengerId']


x, y = load_data('../../data/Titanic.train.csv', True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

lr = LogisticRegression(penalty='l2')
lr.fit(x_train, y_train)
y_hat = lr.predict(x_test)
lr_acc = accuracy_score(y_test, y_hat)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
y_hat = rfc.predict(x_test)
rfc_acc = accuracy_score(y_test, y_hat)

# XGBoost
data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [] #[(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
y_hat = bst.predict(data_test)
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
xgb_acc = accuracy_score(y_test, y_hat)

print('Logistic回归：%.3f%%' % lr_acc)
print('随机森林：%.3f%%' % rfc_acc)
print('XGBoost：%.3f%%' % xgb_acc)