# -*- coding: utf-8 -*-
# @Time    : 2019-06-10 19:48
# @Author  : zxl
# @FileName: main.py

import numpy as np
import pandas as pd
from DT import RT
from RF import RandomForest
from sklearn import tree
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
from GradientBoosting import GradientBoosting

from Util import *
from Bagging import Bagging

if __name__=="__main__":
    root = "D://Data/sale/"
    train_path = root + "train.csv"
    test_path = root + "test.csv"
    out_file=root+"result/result-sklearn-random-forest.csv"


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

    model=GradientBoosting(n_estimators=20)
    model.fit(train_dataset,train_label)
    predict_y=model.predict(test_dataset)





    SaveFile(predict_y,out_file)



