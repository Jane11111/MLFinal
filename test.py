# -*- coding: utf-8 -*-
# @Time    : 2019-06-10 17:53
# @Author  : zxl
# @FileName: test.py

from datetime import datetime

s=datetime.now()
e=datetime.now()

print("pass: %d"%(e-s).seconds)