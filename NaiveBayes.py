# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:22:26 2020

@author: annik
"""

import numpy as np
from scipy.stats import norm
import math
import itertools

class NaiveB:
    def __init__(self): 
        self.N = None 
        self.d = None 
        self.k = None 
        self.cov = None
        self.mean = None
        self.X = None
        self.y = None
        self.mean_array = None
        self.cov_array = None
        self.P_Cj_list = None

    # d,N = np.shape(X)
    # k = np.size(np.unique(y))
    def calc_prior(self):
        #self.k = np.size(np.unique(self.y))
        Nj_list = []
        for i in range(self.k):
            Nj = np.count_nonzero(self.y==i)
            Nj_list.append(Nj)
        # Nj,h = np.shape(X[0, y==0])

        # class priors
        self.P_Cj_list = np.array(Nj_list)/self.N



    # sum all values of dimension n in class j
    # X_d1_c0 = X[0,y==0]
    # X_d1_c1 = X[0,y==1]
    # X_d1_c2 = X[0,y==2]
    # X_d2_c0 = X[1,y==0]
    # X_d2_c1 = X[1,y==1]
    # X_d2_c2 = X[1,y==2]
    
    def calc_mean(self):
        #self.d,self.N = np.shape(self.X)
    #     self.k = np.size(np.unique(self.y))
        # mean of dimension n and class j
        self.mean_array = np.zeros((self.d,self.k))

        for j in range(self.k):
            for n in range(self.d):
                self.mean_array[n,j] = np.mean(self.X[n,self.y==j])
        #     X_c0 = X[d,y==0]
        #     X_c1 = X[d,y==1]
        #     X_c2 = X[d,y==2]



    def calc_cov(self):
        # covariance of dimension n and class j
        self.cov_array = np.zeros((self.d,self.k))

        # calculate covariance
        for j in range(self.k):
            for n in range(self.d):
                self.cov_array[n,j] = np.cov(self.X[n,self.y==j], bias=True)
                #cov_array[n,j] = math.sqrt(np.cov(X[n,y==j]))


    #     cov_array_own = np.zeros((d,k))        

    #attempt to do cov without cov function

    # for j in range(k):
    #        for n in range(d):
    #             minus = X[n,y==j]-mean_array[n,j]
    #             minus_square = minus**2
    #             sums = np.sum(minus_square)
    #             cov_array_own[n][j] = sums/Nj_list[j]

    #print(cov_array)
    #print(cov_array_own)


    # P_0_0 = norm( mean_array[0][0] , np.sqrt( cov_array[0][0] ) ) # normal distribution of class 0's x dimension
    # P_1_0 = norm( mean_array[1][0] , np.sqrt( cov_array[1][0] ) ) # normal distribution of class 0's y dimension
    # def to calc any distr
    def inner_distribution(self,cl,dim):
        return norm( self.mean_array[dim][cl] , np.sqrt( self.cov_array[dim][cl] ) )


    # def P_0(x_sample,y_sample):
    #     return P_0_0.pdf(x_sample)*P_1_0.pdf(y_sample)*P_Cj_list[0]
    # full distr prob of class across all dim
    def P_class(self, cl, data_point): 
        mult = 1;
        for n in range(self.d):
             mult = mult * self.inner_distribution(cl,n).pdf(data_point[n]) 
        return mult * self.P_Cj_list[cl]  

    def arg_max_func(self,data_point):
        P_class_compare_list = []
        for j in range(self.k):
            P_class_compare_list.append(self.P_class(j,data_point))
        return np.argmax(np.array(P_class_compare_list))

    def fit(self, X,y):
        self.d,self.N = X.shape
        self.X = X
        self.y = y
        self.k = np.size(np.unique(self.y))
        self.calc_prior()
        self.calc_mean()
        self.calc_cov()
        
    def predict(self,data_set):
        # y_pred = np.zeros((1,N))
        y_pred_list = []
        for i in range(self.N):
            y_pred_list.append(self.arg_max_func(data_set[:,i]))
        return y_pred_list
