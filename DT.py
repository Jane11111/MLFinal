# -*- coding: utf-8 -*-
# @Time    : 2019-06-10 14:36
# @Author  : zxl
# @FileName: DT.py

"""
实现一棵回归树 CART
"""
import numpy as np

from sklearn.metrics import mean_squared_error

class RT:

    def __init__(self):
        pass

    def GenerateTree(self,dataset,label):

        assert dataset is not None

        (feature,value)=self.ChooseBestSplit(dataset,label)

        if feature==None:#说明label全都一样，或者feature上取值全都一样
            return value
        """
        如果说feature不为None，那么就是能够被正常划分了。
        是不是还存在着，把数据全都划到一边能够使得MSE变小？如果这样的话就是无限迭代了，需要避免该情况发生
        如果说不管怎么切分，都不会使得MSE变小，那么就直接不进行划分了，同样返回（None，value）
        """
        tree={}
        tree['feat']=feature
        tree['val']=value
        (left_dataset, left_label, right_dataset, right_label)=self.SplitData(dataset,label,feature,value)
        tree['left']=self.GenerateTree(left_dataset,left_label)
        tree['right']=self.GenerateTree(right_dataset,right_label)
        return tree

    def ChooseBestSplit(self,dataset,label,candidate_fea=None):

        assert dataset is not None

        feature=None
        value=None
        mean_value=np.mean(label)
        MSE=mean_squared_error(label,np.full(shape=(len(label),),fill_value=mean_value))
        if MSE==0:#所有的label都一样,可以结束了，不再划分
            return (None,mean_value)

        if candidate_fea is None:
            candidate_fea=range(len(dataset[0]))
        for i in candidate_fea:
            split_value=np.mean(dataset[:,i])
            (left_dataset, left_label, right_dataset, right_label) =self.SplitData(dataset,label,i,split_value)
            if left_dataset is None or right_dataset is None: #说明在该feature上，所有取值都一样
                continue
            left_mean=np.mean(left_label)
            right_mean=np.mean(right_label)
            left_MSE=mean_squared_error(left_label,np.full(shape=(len(left_label),),fill_value=left_mean))
            right_MSE=mean_squared_error(right_label,np.full(shape=(len(right_label),),fill_value=right_mean))
            total_MSE=(left_MSE*len(left_label)+right_MSE*len(right_label))/len(label)
            if total_MSE<MSE:#得到了一个更小的MSE值，可以从改点开始划分
                MSE=total_MSE
                feature=i
                value=split_value

        if feature is None:
            value=mean_value

        return (feature,value)


    def SplitData(self,dataset,label,feature,value):

        left_index=np.argwhere(dataset[:,feature]<=value).flatten()
        right_index=np.argwhere(dataset[:,feature]>value).flatten()

        if len(left_index)==0:
            left_dataset=None
            left_label=None
        else:
            left_index=left_index
            left_dataset = dataset[left_index,:]
            left_label = label[left_index]

        if len(right_index)==0:
            right_dataset=None
            right_label=None
        else:
            right_index=right_index
            right_dataset = dataset[right_index,:]
            right_label = label[right_index]

        return (left_dataset,left_label,right_dataset,right_label)

    def fit(self,dataset,label):
        self.dataset = dataset
        self.label = label
        self.tree=self.GenerateTree(self.dataset,self.label)

    def predict(self,dataset):
        labels=[]
        for x in dataset:
            val=self.predict_singlevalue(x)
            labels.append(val)
        return labels


    def RecursivePredict(self,tree,x):
        if type(tree) is dict:
            fea=tree['feat']
            val=tree['val']
            x_i=x[fea]
            if x_i<=val:
                return self.RecursivePredict(tree['left'],x)
            else:
                return self.RecursivePredict(tree['right'],x)
        else:
            return tree


    def predict_singlevalue(self,x):

        res=self.RecursivePredict(self.tree,x)
        return res



# if __name__ == "__main__":
#
#     dataset=np.array([[1,2,3],
#                       [2,3,4],
#                       [3,4,5],
#                       [5,6,7],
#                       [6,7,8],
#                       [7,8,9]])
#     label=np.array([1,2,3,4,5,6])
#
#     test_dataset=np.array([[1,2,3],
#                            [2,3,4],
#                            [7,8,9]])
#     model=RT(dataset,label)
#     model.fit()
#     y=model.predict(test_dataset)
#     print(y)
#
