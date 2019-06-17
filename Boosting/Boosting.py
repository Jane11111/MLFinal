# -*- coding: utf-8 -*-
# @Time    : 2019-06-16 18:04
# @Author  : zxl
# @FileName: Boosting.py

"""
这个文件将boosting方法进行封装
后面实现的，只需要调整一下 y还有w的更新函数就行了

"""

import numpy as np
from Single.DT import RT

class Boosting():
    def __init__(self,n_estimators=20):
        self.n_estimators=n_estimators

    def update_y(self,dataset,label,last_y):
        direction_y = last_y - label
        model = RT()
        model.fit(dataset, direction_y)
          # 首先找到一个向梯度降低方向学习的model
        cur_predict_y = model.predict(dataset)  # 是hm(x)

        if np.sum(cur_predict_y) == 0:
            weight = 0
        else:
            weight = np.sum(label - last_y) / np.sum(cur_predict_y)
        last_y = last_y + weight * cur_predict_y
        return model,last_y,weight


    def fit(self,dataset,label):

        model_list=[]
        weight_list=[]
        last_y=np.full(shape=(len(label),),fill_value=np.mean(label))#是F m-1(x)
        model_list.append(np.mean(label))
        weight_list.append(1)
        for i in range(self.n_estimators):

            model,last_y,weight=self.update_y(dataset,label,last_y)
            model_list.append(model)
            weight_list.append(weight)

        self.model_list=model_list
        self.weight_list=weight_list


    def predict(self,dataset):
        res=np.full(shape=(len(dataset),),fill_value=self.model_list[0])
        for i in np.arange(1,self.n_estimators+1,1):
            cur_predict_y=self.model_list[i].predict(dataset)
            res+=cur_predict_y*self.weight_list[i]
        return res
