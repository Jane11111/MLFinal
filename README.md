# 机器学习课程期末项目

## 任务
  
  给出一系列房子的特征，预测房子价格。
  
## 数据集

### 地址

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description

### 预处理

由于数据中含有较多空值，因此首先对空值进行填充，填充的值为该特征的众数，在将空值填充完成以后，对离散值使用one-hot方式进行处理。最终每个房子由一个特征向量组成，房价为预测目标。

### 训练&测试数据

训练集共1460条数据，共有79个特征，35个数值型特征，44个离散型特征。SalePrice为预测标签。

测试集共1459条数据，特征同上，不含标签。


## 算法
  
  回归树：Single/DT.py
  
  多层感知机: Single/MLP.py
  
  Bagging：Bagging/Bagging.py
  
  随机森林：Bagging/RF.py
  
  Gradient Boosting：
  
  平方损失：GradientBoosting.py
  
  绝对值损失：LAD_Boost.py
  
## 其他文件

Tool/Util.py: 包含预处理代码，以及bootstrap方法等。

## 运行
  
  只需要将main.py文件中对应的方法注释去掉，即可运行出结果。


