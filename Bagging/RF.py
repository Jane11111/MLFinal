# -*- coding: utf-8 -*-
# @Time    : 2019-06-11 10:08
# @Author  : zxl
# @FileName: RF.py

from Bagging.Bagging import Bagging
from Single.RandomTree import RandomTree
from Tool.Util import *
"""
实现随机森林方法
"""
class RandomForest(Bagging):

    def __init__(self,baselearner_num):
        super().__init__(baselearner_num)


    def fit(self,dataset,label):
        self.dataset = dataset
        self.label = label

        percent = 0.8
        for i in range(self.num):
            (sample_dataset, sample_label) = Bootstrap(self.dataset, self.label, percent)
            model = RandomTree()
            model.fit(sample_dataset, sample_label)
            self.models.append(model)

