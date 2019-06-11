# -*- coding: utf-8 -*-
# @Time    : 2019-06-10 17:53
# @Author  : zxl
# @FileName: test.py

import numpy as np

l1=np.array([[1],[1],[1]])
l2=np.array([2,2,2])
l2=l2[:,np.newaxis]

l3=np.hstack((l1,l2,l2))
l4=l3[:,:-1]
l5=l3[:,-1]
print(l4)
print(l5)