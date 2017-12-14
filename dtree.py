from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
import numpy as np


data = pd.read_csv('train_aWnotuB.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
test = pd.read_csv('test_BdBKkAj.csv')


def process(data):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    for time_name in ['hour', 'day', 'week', 'month', 'weekday', 'year']:
        cal_string = ''.join(['x.', time_name])
        if time_name == 'weekday':
            cal_string = ''.join(['x.', time_name, '()'])
        data[time_name] = data['DateTime'].apply(lambda x: eval(cal_string))
    return data


def dummy(data):
    X = pd.DataFrame()
    for col_name in ['Junction', 'day']:
        X = pd.concat([X, pd.get_dummies(data[col_name], prefix=col_name)], axis=1)
    X = pd.concat([X, data[['hour','week', 'weekday', 'month']], np.log(data['year'])], axis=1)
    return X


def past_data(data, shift_hour=0):
    X2 = pd.DataFrame()
    for i in range(shift_hour):
        X2[''.join(['shift_', str(i+1)])] = data['Vehicles'].shift(i+1)
    return X2


def transform_data(data, shift_hour=0):
    process_data = process(data)
    X_dummy = dummy(process_data)
    pastdata = past_data(process_data)
    X_train = pd.concat([X_dummy,pastdata],axis=1)[shift_hour:]
    return X_train


def get_test(test, data, junction=1, shift_hour=0):
    process_test = process(test)
    X_dummy = dummy(process_test)

    # 选择一个junction 做test
    col_name = ''.join(['Junction_', str(junction)])
    X_test = X_dummy[X_dummy[col_name] == 1]

    X_test.index = range(len(X_test))

    # 找这个junction最后5天的数据
    past_5 = data[data['Junction'] == junction][-shift_hour:]['Vehicles'].values
    column = [''.join(['shift_', str(i+1)]) for i in range(shift_hour)]
    all = dict(zip(column, past_5))
    pastdata = pd.DataFrame(pd.Series(all,)).T
    X_test = pd.concat([X_test,pastdata],axis=1)
    return X_test


def get_result(test, data, model, shift_hour=0):
    y_test = []
    column = [''.join(['shift_', str(i+1)]) for i in range(shift_hour)]
    # for i in [1, 2, 3, 4]:
    for i in [1, 2, 3, 4]:
        X_test = get_test(test, data, junction=i)
        data_i = data[data['Junction'] == i]
        diff_size = data_i['Vehicles'].diff().mean()*0.3

        for j in range(len(X_test)):

            y = model.predict(X_test[j:j + 1])[0] + diff_size * (j+2)
            y_test.append(y)
            # 更新shift
            # past_5 = X_test[j:j+1][column]
            # if j == 2951:
            #     break
            # new_past = past_5.T.shift(1).fillna(y).T
            # new_past.index = X_test[j+1:j+2].index
            # X_test.loc[j+1:j+2, column] = new_past

    y = pd.DataFrame(y_test, columns=['Vehicles'])
    result = pd.concat([test[['DateTime', 'Junction', 'ID']], y], axis=1)
    result.to_csv('result.csv')

shift_hour = 0

y_train = data['Vehicles'][shift_hour:]
X_train = transform_data(data)


# 线性回归
# model = LinearRegression()
# model.fit(np.array(X_train), np.array(y_train))
# model.score(np.array(X_train), np.array(y_train))
# get_result(test, data, model)

# 决策树

# parameters = {'criterion': ['friedman_mse'], 'min_samples_split':np.arange(2,10,1), 'min_samples_leaf':np.arange(2,10,1)}
# dtree = DecisionTreeRegressor()
# model2 = GridSearchCV(dtree, parameters,cv=1)
# model2.fit(np.array(X_train), np.array(y_train))


model2 = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=5, min_samples_leaf=5, presort=True)
# model2 = GradientBoostingRegressor()
model2.fit(np.array(X_train), np.array(y_train))


# get_result(test, data, model2)

import pydotplus
import graphviz
from sklearn.tree import export_graphviz
dot = export_graphviz(model2, out_file=None)
# with open('treemodel.txt','w') as f:
#     f.write(dot)
# graphviz.render('dot', 'png', 'treemodel.txt')
tree_plot = pydotplus.graph_from_dot_data(dot)
tree_plot.write_pdf('treeplot.pdf')
