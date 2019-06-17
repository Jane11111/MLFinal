# -*- coding: utf-8 -*-
# @Time    : 2019-06-17 08:46
# @Author  : zxl
# @FileName: LAD_Boost.py

"""
这个文件实现了以|y-f(x)|作为损失函数的gradient-boosting方法
"""

import numpy as np
from Single.DT import RT
from Boosting import Boosting

class LAD_Boost(Boosting):

    def __init__(self, n_estimators=20):
        super(Boosting, self).__init__(n_estimators)

    def update_y(self, dataset, label, last_y):

        direction_y = np.sign( label -last_y ) # 也即 y-波浪
        model = RT()
        model.fit(dataset, direction_y)  # 首先找到一个向梯度降低方向学习的model

        cur_predict_y = model.predict(dataset)  # 是hm(x)
        new_list=[]
        for i in range(len(cur_predict_y)):
            w=np.abs(cur_predict_y[i])
            r=label[i]-last_y[i]
            if cur_predict_y[i]==0:
                r=0
            else:
                r=r/cur_predict_y[i]
            new_list.append(w*r)
        weight=np.median(new_list)
        last_y = last_y + weight * cur_predict_y
        return model, last_y, weight


