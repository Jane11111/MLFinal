# -*- coding: utf-8 -*-
# @Time    : 2019-06-10 19:48
# @Author  : zxl
# @FileName: main.py

from Boosting.LAD_Boost import LAD_Boost
from Boosting.GradientBoosting import GradientBoosting
from Bagging.Bagging import Bagging
from Bagging.RF import RandomForest
from Single.DT import RT
from Single.MLP import MLP

from datetime import datetime
from Tool.Util import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor,RandomForestRegressor
from sklearn import tree

if __name__=="__main__":
    root = "D://Data/sale/"
    train_path = root + "train.csv"
    test_path = root + "test.csv"
    out_file=root+"result/result-sklearn-test.csv"


    train_dataset,train_label,test_dataset=LoadData(train_path,test_path)
    s=datetime.now()

    #model=RT()
    #model.fit(train_dataset,train_label)
    #predict_y=model.predict(test_dataset)

    model=tree.DecisionTreeRegressor(random_state=1)
    predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    #model=Bagging(20)
    #model.fit(train_dataset,train_label)
    #predict_y=model.predict(test_dataset)
    #model=BaggingRegressor(n_estimators=20)
    #predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    #model = RandomForest(20)
    #model.fit(train_dataset,train_label)
    #predict_y = model.predict(test_dataset)
    #model=RandomForestRegressor(n_estimators=20)
    #predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    #model=GradientBoosting(n_estimators=20)
    #model.fit(train_dataset,train_label)
    #predict_y=model.predict(test_dataset)
    #model=GradientBoostingRegressor(n_estimators=20)
    #predict_y=model.fit(train_dataset,train_label).predict(test_dataset)

    #model=LAD_Boost(n_estimators=20)
    #model.fit(train_dataset,train_label)
    #predict_y=model.predict(test_dataset)

    #model=MLP(hc=5,beth=0.001,epoch=500,h_activation="None",o_activation="None")
    ##model=MLPRegressor(hidden_layer_sizes=5,activation="identity",max_iter=500)
    #scaler = MinMaxScaler(feature_range=(0,1))
    #train_dataset=scaler.fit_transform(train_dataset)
    #m1=min(train_label)
    #m2=max(train_label)
    #train_label=(train_label-min(train_label))/(max(train_label)-min(train_label))
    #model.fit(train_dataset,train_label)
    #test_dataset=scaler.fit_transform(test_dataset)
    #predict_y=model.predict(test_dataset)
    #predict_y=predict_y*(m2-m1)+m1

    e=datetime.now()
    print("pass: %d ms"%((e-s).microseconds/1000))



    SaveFile(predict_y,out_file)



