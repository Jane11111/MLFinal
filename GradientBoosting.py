# -*- coding: utf-8 -*-
# @Time    : 2019-06-11 17:39
# @Author  : zxl
# @FileName: GradientBoosting.py


import numpy as np
from DT import RT

class GradientBoosting():
    def __init__(self,n_estimators=20):
        self.n_estimators=n_estimators

    def fit(self,dataset,label):

        model_list=[]
        weight_list=[]
        last_y=np.full(shape=(len(label),),fill_value=np.mean(label))#是F m-1(x)
        model_list.append(np.mean(label))
        weight_list.append(1)
        for i in range(self.n_estimators):
            direction_y=last_y-label
            model=RT()
            model.fit(dataset,direction_y)
            model_list.append(model)#首先找到一个向梯度降低方向学习的model
            cur_predict_y=model.predict(dataset) #是hm(x)
            if np.sum(cur_predict_y)==0:
                weight=0
            else:
                weight=np.sum(label-last_y)/np.sum(cur_predict_y)
            weight_list.append(weight)
            last_y=last_y+weight*cur_predict_y
        self.model_list=model_list
        self.weight_list=weight_list


    def predict(self,dataset):
        res=np.full(shape=(len(dataset),),fill_value=self.model_list[0])
        for i in range(self.n_estimators):
            cur_predict_y=self.model_list[i].predict(dataset)
            res+=cur_predict_y*self.weight_list[i]
        return res