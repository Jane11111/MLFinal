# -*- coding: utf-8 -*-
# @Time    : 2019-06-10 19:48
# @Author  : zxl
# @FileName: main.py

from sklearn.ensemble import GradientBoostingRegressor
from Boosting.LAD_Boost import LAD_Boost
from Boosting.GradientBoosting import GradientBoosting
from Bagging.Bagging import Bagging
from Bagging.RF import RandomForest
from Single.DT import RT
from Single.MLP import MLP
from Tool.Util import *
from sklearn.preprocessing import MinMaxScaler

if __name__=="__main__":
    root = "~/Documents/数据集/sale/"
    train_path = root + "train.csv"
    test_path = root + "test.csv"
    out_file=root+"result/result-mlp.csv"


    train_dataset,train_label,test_dataset=LoadData(train_path,test_path)

    # model=RT()
    # model.fit(train_dataset,train_label)
    # predict_y=model.predict(test_dataset)

    # model=tree.DecisionTreeRegressor(random_state=1)
    # predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    # model=Bagging(20)
    # model.fit(train_dataset,train_label)
    # predict_y=model.predict(test_dataset)
    # model=BaggingRegressor(n_estimators=20)
    # predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    # model = RandomForest(20)
    # model.fit(train_dataset,train_label)
    # predict_y = model.predict(test_dataset)
    # model=RandomForestRegressor(n_estimators=20)
    # predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    # model=GradientBoosting(n_estimators=20)
    # model.fit(train_dataset,train_label)
    # predict_y=model.predict(test_dataset)
    # model=GradientBoostingRegressor(n_estimators=20)
    # predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    # model=LAD_Boost(n_estimators=20)
    # model.fit(train_dataset,train_label)
    # predict_y=model.predict(test_dataset)

    model=MLP(hc=10,beth=0.01,epoch=50,h_activation="tanh",o_activation="tanh")
    scaler = MinMaxScaler(feature_range=(0,1))
    train_dataset=scaler.fit_transform(train_dataset)
    m1=min(train_label)
    m2=max(train_label)
    train_label=(train_label-min(train_label))/(max(train_label)-min(train_label))
    model.fit(train_dataset,train_label)
    test_dataset=scaler.fit_transform(test_dataset)
    predict_y=model.predict(test_dataset)
    predict_y=predict_y*(m2-m1)+m1



    SaveFile(predict_y,out_file)



