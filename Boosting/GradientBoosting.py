# -*- coding: utf-8 -*-
# @Time    : 2019-06-11 17:39
# @Author  : zxl
# @FileName: GradientBoosting.py


import numpy as np
from Single.DT import RT
from Boosting import Boosting

class GradientBoosting(Boosting):
    def __init__(self,n_estimators=20):
        super(Boosting,self).__init__(n_estimators)

    def update_y(self,dataset,label,last_y):

        direction_y = last_y - label #也即 y-波浪
        model = RT()
        model.fit(dataset, direction_y)# 首先找到一个向梯度降低方向学习的model

        cur_predict_y = model.predict(dataset)  # 是hm(x)

        if np.sum(cur_predict_y) == 0:
            weight = 0
        else:
            weight = np.sum(label - last_y) / np.sum(cur_predict_y)
        last_y = last_y + weight * cur_predict_y
        return model,last_y,weight

