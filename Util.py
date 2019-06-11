# -*- coding: utf-8 -*-
# @Time    : 2019-06-11 10:11
# @Author  : zxl
# @FileName: Util.py

import numpy as np
import pandas as pd

"""
这个文件里面存放一些常用的方法
"""

def Bootstrap(dataset,label,percent):
    label=label[:,np.newaxis]

    full_dataset=np.hstack((dataset,label))

    df=pd.DataFrame(full_dataset)
    samples=np.array(df.sample(frac=percent,replace=True))

    sample_dataset=samples[:,:-1]
    sample_label=samples[:,-1]

    return (sample_dataset,sample_label)


def SaveFile(predict_y,file_path):
    res = {}
    ids = []
    predicts = []
    for i in range(len(predict_y)):
        ids.append(i + 1461)
        predicts.append(predict_y[i])
    res['Id'] = ids
    res['SalePrice'] = predicts
    df = pd.DataFrame(res)
    df.to_csv(file_path, index=False)

def LoadData(train_path,test_path):


    train_df=pd.read_csv(train_path,usecols=np.arange(1,80,1))


    for index, row in train_df.iteritems():
        v=train_df[index].mode()[0]
        train_df[index]=train_df[index].fillna(v)
    raw_train_dataset=train_df.values

    train_df=pd.read_csv(train_path,usecols=[80])
    train_label=(train_df.values).flatten()

    test_df=pd.read_csv(test_path,usecols=np.arange(1,80,1))
    for index, row in test_df.iteritems():
        test_df[index]=test_df[index].fillna(test_df[index].mode()[0])
    raw_test_dataset=test_df.values

    """
    将string类别变成one-hot表示
    """
    category_map={}
    for i in range(len(raw_train_dataset[0])):
        if type(raw_train_dataset[0][i]) is not str:
            continue
        category_map[i]={}
        train_col=raw_train_dataset[:,i]
        test_col=raw_test_dataset[:,i]
        cur_col=list(set(np.append(train_col,test_col)))

        for j in range(len(cur_col)):
            arr=np.full(shape=(len(cur_col,)),fill_value=0)
            arr[j]=1
            category_map[i][cur_col[j]]=arr


    train_dataset=[]
    for i in range(len(raw_train_dataset)):
        cur_row=[]
        for j in range(len(raw_train_dataset[i])):
            val=raw_train_dataset[i][j]
            if type(val) is str:
                cur_row.extend(category_map[j][val].tolist())
            else:
                cur_row.append(val)
        train_dataset.append(cur_row)
    train_dataset=np.array(train_dataset)

    test_dataset=[]
    for i in range(len(raw_test_dataset)):
        cur_row=[]
        for j in range(len(raw_test_dataset[i])):
            val=raw_test_dataset[i][j]
            if type(val) is str:
                cur_row.extend(category_map[j][val].tolist())
            else:
                cur_row.append(val)
        test_dataset.append(cur_row)
    test_dataset=np.array(test_dataset)

    return (train_dataset,train_label,test_dataset)




