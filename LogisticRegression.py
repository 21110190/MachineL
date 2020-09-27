# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:24:11 2020

@author: annik
"""

import numpy as np
from scipy.stats import norm
import math
import itertools

class LogisticReg:
    
    def ___init___(self):
#         self.lambd = 2
        self.X = None
        self.y = None
        self.N = None
        self.k = None
        self.d = None
        self.W_fin = None
        

    # sigmoid function
    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

#test sigmoid
#X = np.array([-1,2,-3,4])
#print(sigmoid(X))

#E(w) = -logp(D|w)
# def cost(X,y,wt):
#     d,N = np.shape(X) 
#     ans = 0
#     for i in range(self.N):
#         sig = sigmoid(np.dot(wt,X[:,i]))
#         ans = ans + (-y[0,i] * np.log(sig) - (1-y[0,i]) * np.log(1-sig))        
#     return ans

    def cost(self, W):
#         d,N = np.shape(X)
        ans = 0
        Wt = W.reshape((1,2))
        for n in range(self.N):
            sig = self.sigmoid(np.dot(Wt, self.X[:,n]))
            ans = ans + self.y[n] * np.log(sig) + (1-self.y[n]) * np.log(1-sig)  
        return -ans + (1/(2*2))*Wt*W

# #calc gradient
    def gradient(self, W):
        #X = np.matrix([[-1,2,7],[-3,0,5]])
        #y = np.array([0,0,1])
        Wt = W.reshape((1,2))
#         d,N = np.shape(X)
        grad = np.matrix([[0],[0]])

        for n in range(self.N):
            x = self.X[:,n]
            xt = np.reshape(x, (2, 1))
            sig = self.sigmoid(np.dot(Wt, xt))
            sig1 = self.y[n]-sig
            grad = grad + (float(sig1))*xt 

            # grad = grad + np.dot((y[n]-sig),X[:,n])
        print(-grad + (1)*W)
        return -grad + (1)*W

    # #calculate Hessian
    def hessian(self, W):
        #X = np.matrix([[-1,2,7],[-3,0,5]])
        #XT = X.T
        #y = np.matrix([[0,0,1]])
        Wt = W.reshape((1,2))
#         d,N = np.shape(X)
        hes = np.zeros((self.d,self.d))


        for n in range(self.N):
            x = self.X[:,n].reshape((2,1))
            xt = np.reshape(x, (1, 2))
            sig = self.sigmoid(np.dot(Wt,x))
            hes = hes + float(sig*(1-sig))*np.dot(x,xt)
             
        return hes + (0.5*np.diag([1, 1]))

    #update w using newtons method => wk+1 = wk - (grad/hes)
    def updateW(self, W):
        hessianInv = np.linalg.inv(self.hessian(W)) 
        grad = self.gradient(W)
        dot = np.dot(hessianInv, grad)
        W_new = W - dot
        #Wt_new = W_new.T
        #Wt = Wt - np.dot(hessianInv, grad)
        return W_new


    def fit(self, X, y):
        self.d,self.N = X.shape
        self.X = X
        self.y = y
        
        w_old = np.matrix([[0.5],[1]])
        perc = 1
        while (perc > 0.01):
            w_new = self.updateW(w_old)
            perc = np.linalg.norm(w_old - w_new)/np.linalg.norm(w_old)
            w_old = w_new
            self.W_fin = w_new
            
        
        
    
    def predict(self):
        y_pred = []
        Wt = self.W_fin.reshape((1,2))
        for n in range(self.N):
            class_assign = 0
            if self.sigmoid(np.dot(Wt,self.X[:,n].reshape((2,1))))>0.5:
                    class_assign = 1
            y_pred.append(class_assign)

        return np.array(y_pred)
