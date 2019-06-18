# -*- coding: utf-8 -*-
# @Time    : 2019-06-17 17:34
# @Author  : zxl
# @FileName: MLP.py


import random
import numpy as np
np.random.seed(153)

class MLP:

    """
    hc,公有特征对应隐层个数
    h_activation:隐层激活函数
    o_activation:输出层激活函数
    """
    def __init__(self,hc,beth,epoch=50,h_activation="sigmod",o_activation="sigmod"):
        self.hc=hc
        self.beth=beth
        self.epoch=epoch
        self.h_activation=h_activation
        self.o_activation=o_activation

    """
    y：标签
    """
    def fit(self,X,y):
        #这是两部分输入层的单元个数
        self.i1=len(X[0])
        #TODO,以下的随机数都应该应该是以随机数填充
        Vc=np.random.random((self.i1,self.hc))
        W=np.random.random((self.hc,))#是个向量
        theta=random.random()
        gama_c=np.random.random((self.hc,))#一个向量


        ###三角读作delta

        """
        开始遍历+更新
        """
        epoch=self.epoch
        while epoch>0:
            print("epoch: %d"%(self.epoch-epoch))

            epoch-=1
            loss=0
            total=0
            for i in range(len(X)):
                real_y=y[i]
                # 隐层输入
                alpha_c = np.dot(X[i],Vc )
                alpha=alpha_c
                # 隐层输出
                bc_vec = alpha_c - gama_c
                for j in range(len(bc_vec)):
                    bc_vec[j] = self.Activation(bc_vec[j], self.h_activation)

                h = bc_vec
                # 输出层输入
                beta = np.dot(W, h)
                # 输出层输出
                pred_y = self.Activation(beta-theta, self.o_activation)


                loss+=pow(pred_y-real_y,2)
                total+=1

                g=(real_y-pred_y)*self.Derivation(beta-theta,self.o_activation)
                delta_theta=-self.beth*g
                delta_W=self.beth*g*h

                gama_vec=gama_c
                e_vec=g*W

                for j in range(len(e_vec)):
                    e_vec[j]=e_vec[j]*self.Derivation(alpha[j]-gama_vec[j],self.h_activation)
                #print(e_vec)

                delta_gama_c=-self.beth*e_vec

                delta_Vc=np.full(shape=Vc.shape,fill_value=0,dtype=np.float32)
                for j in range(len(delta_Vc)):
                    delta_Vc[j]=X[i][j]*e_vec#一行一行地求导

                delta_Vc*=self.beth

                Vc+=delta_Vc

                W+=delta_W
                theta+=delta_theta
                gama_c+=delta_gama_c
            loss/=(2*total)
            print("loss: %f"%(loss))

        self.Vc=Vc
        self.W=W
        self.theta=theta
        self.gama_c=gama_c


    """
    在网络结构已定的情况下，预测y
    """
    def predict_y(self,x):
        #隐层输入
        alpha_c=np.dot(x,self.Vc)
        #隐层输出
        bc_vec=alpha_c-self.gama_c
        for i in range(len(bc_vec)):
            bc_vec[i]=self.Activation(bc_vec[i],self.h_activation)

        h=bc_vec
        #输出层输入
        beta=np.dot(self.W,h)-self.theta
        #输出层输出
        res=self.Activation(beta,self.o_activation)
        return (h,res)



    def predict(self, X):
        res=[]
        for i in range(len(X)):
            (h,r)=self.predict_y(X[i])
            res.append(r)
        return np.array(res)



    """
    求导
    """
    def Derivation(self,c,activation):
        if activation=="sigmod":
            y=1/(1+pow(np.e,-c))
            res=y*(1-y)
        elif activation=="tanh":
            res=1-pow(np.tanh(c),2)
        elif activation=="relu":
            if c < 0:
                res = 0
            else:
                res = 1
        else:
            res=1
        return res
    def Activation(self,c,activation):
        if activation=="sigmod":
            res=1/(1+pow(np.e,-c))
        elif activation=="tanh":
            res=np.tanh(c)
        elif activation=="relu":
            res=max(0,c)
        else:
            res=c
        return res





