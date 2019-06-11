# -*- coding: utf-8 -*-
# @Time    : 2019-06-11 10:08
# @Author  : zxl
# @FileName: Bagging.py

from Util import *
from DT import RT

class Bagging:

    """
    baselearner_num:基学习器的数量
    """
    def __init__(self,baselearner_num):
        self.num=baselearner_num
        self.models=[]

    def fit(self,dataset,label):
        self.dataset=dataset
        self.label=label

        percent=0.8
        for i in range(self.num):
            (sample_dataset,sample_label)=Bootstrap(self.dataset,self.label,percent)
            model=RT()
            model.fit(sample_dataset,sample_label)
            self.models.append(model)

    def predict(self,dataset):
        res_m=[]
        for model in self.models:
            cur_res=model.predict(dataset)
            res_m.append(cur_res)
        res=self.merge(np.array(res_m))
        return res

    """
    res_m 的每一行对应一个基学习器学到的结果
    """
    def merge(self,res_m):
        res=np.mean(res_m,axis=1)
        return res



