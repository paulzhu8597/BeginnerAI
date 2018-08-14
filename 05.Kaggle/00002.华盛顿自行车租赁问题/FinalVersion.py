import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
from matplotlib import pyplot as plt

DATA_PATH = "../input/"

train_data_frame = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_data_frame = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

# prices = pd.DataFrame({"count":train_data_frame["count"], "log(Count + 1)":np.log1p(train_data_frame["count"])})
# prices.hist()
# plt.show()
def data_preprocessing(dataFrame):
    scaler = StandardScaler()
    dataFrame['month'] = pd.DatetimeIndex(dataFrame['datetime']).month
    dataFrame['day'] = pd.DatetimeIndex(dataFrame['datetime']).dayofweek
    dataFrame['hour'] = pd.DatetimeIndex(dataFrame['datetime']).hour
    dataFrame.drop(['datetime'], axis=1, inplace=True)

    dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame['season'], prefix='season')], axis=1)
    dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame['holiday'], prefix='holiday')], axis=1)
    dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame['workingday'], prefix='workingday')], axis=1)
    dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame['weather'], prefix='holiday')], axis=1)

    temp_scale_param = scaler.fit(dataFrame['temp'].values.reshape(-1,1))
    dataFrame['temp_scaled'] = scaler.fit_transform(dataFrame['temp'].values.reshape(-1,1), temp_scale_param)

    atemp_scale_param = scaler.fit(dataFrame['atemp'].values.reshape(-1,1))
    dataFrame['atemp_scaled'] = scaler.fit_transform(dataFrame['atemp'].values.reshape(-1,1), atemp_scale_param)

    humidity_scale_param = scaler.fit(dataFrame['humidity'].values.reshape(-1,1))
    dataFrame['humidity_scaled'] = scaler.fit_transform(dataFrame['humidity'].values.reshape(-1,1), humidity_scale_param)

    windspeed_scale_param = scaler.fit(dataFrame['windspeed'].values.reshape(-1,1))
    dataFrame['windspeed_scaled'] = scaler.fit_transform(dataFrame['windspeed'].values.reshape(-1,1), windspeed_scale_param)

    dataFrame.drop(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'], axis=1, inplace=True)
    return dataFrame

def rmsle(y_hat, y):
    np.log(y_hat + 1) - np.log(y + 1)

if __name__ == '__main__':
    train_df = data_preprocessing(train_data_frame)
    train_df.drop(['casual','registered'], axis=1, inplace=True)

    train_y = train_df['count'].values
    train_x = train_df.drop(['count'], axis=1).values

    x_train, x_cv, y_train, y_cv = train_test_split(train_x, train_y, test_size=0.2, random_state=0)

    param_list = {"max_features" : [.1, .3, .5, .7, .9, .99],
                  "n_estimators" : [10, 50, 100, 300, 500, 800]}

    model = GridSearchCV(GradientBoostingRegressor(), cv=5, param_grid=param_list)
    model.fit(train_x, train_y)
    print(model.best_params_)
    print(model.best_score_)

    # test_df = data_preprocessing(test_data_frame)
    # test_x = test_df.values
    # Y = model.predict(test_x)
    #
    # dataFrame = pd.read_csv(os.path.join(DATA_PATH, "00002", "test.csv"))
    # datetime = dataFrame['datetime'].values
    #
    # data = pd.DataFrame({"datetime" : datetime, "count" : Y.astype(np.int32)})
    # data.to_csv("Result.csv", index=False)


