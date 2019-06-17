# -*- coding: utf-8 -*-
# @Time    : 2019-06-11 11:57
# @Author  : zxl
# @FileName: RandomTree.py

import random
import numpy as np
from Single.DT import RT

class RandomTree(RT):
    def __init__(self,random_state=1):
        super(RT,self).__init__()
        np.random.seed(random_state)

    def GenerateRandomTree(self,dataset,label,k="auto"):
        assert dataset is not None

        #TODO 选择候选的特征集合
        if k=="auto":
            k=int(np.sqrt(len(dataset[0])))


        candidate=random.sample(range(len(dataset[0])),k)

        (feature, value) = self.ChooseBestSplit(dataset, label,candidate)

        if feature == None:  # 说明label全都一样，或者feature上取值全都一样
            return value

        tree = {}
        tree['feat'] = feature
        tree['val'] = value
        (left_dataset, left_label, right_dataset, right_label) = self.SplitData(dataset, label, feature, value)
        tree['left'] = self.GenerateTree(left_dataset, left_label)
        tree['right'] = self.GenerateTree(right_dataset, right_label)
        return tree

    def fit(self,dataset,label):
        self.tree=self.GenerateRandomTree(dataset,label)


