# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:29:39 2020

@author: annik
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import HTML
from IPython.display import display
from IPython.display import Image

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

from numpy.random import randint
from skimage import io
from ipywidgets import interact
from matplotlib.patches import Ellipse

# Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
np.random.seed(0)
# Number of points
N = 1000
# Labels for each cluster
y = np.random.randint(low=0, high=2, size = N)
#print(y)
c = np.array(["r","g"])
# Mean of each cluster
means = np.array([[-2, 2], [-2, 2],])
# Covariance (in X and Y direction) of each cluster
covariances = np.random.random_sample((2, 2)) + 1
# Dimensions of each point
X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
               np.random.randn(N)*covariances[1, y] + means[1, y]])

for k in range(X.shape[1]):
    plt.plot(X[0,k],X[1,k],c[y[k]]+"o")
    
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Class labels given")
plt.show()

for k in range(X.shape[1]):
    plt.plot(X[0,k],X[1,k],"m"+"o")
    
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("No Class labels")
plt.show()

# check format of data
#print(X)
#print(X.shape)
#print(X[:,0].reshape(2,1))

# k = 3
# c = np.array(["r","g","b"])
k = 2
d,N = X.shape
T = np.zeros((k,N))
#print(T)

# choose initial means
mu_array = np.zeros((d,k))
#print(mu_array)
#print(T)


# mu_array[0] = [1,1.7,3]
# mu_array[1] = [1,1.5,3]
mu_array[0] = [1,1.7]
mu_array[1] = [1,1.5]

# mu_array[0][0] = 1
# mu_array[0][1] = 0
# mu_array[1][0] = 1
# mu_array[1][1] = 0
# mu_array = np.ones((d,k))
# print(mu_array)

# print(mu_array[:,0])

# first step is to choose initial mean and minimize J w.r.t. mean 
def expectation():
    for n in range(N):
        dist = np.zeros(k)
        for j in range(k):
            T[j][n]=0
            dist[j] = np.linalg.norm(X[:,n] - mu_array[:,j])**2
        T[np.argmin(dist)][n] = 1   #Each data value xn is assigned to the cluster for dist is smallest
   # print(T)
   # print(dist)
    
# second step is to update mean by minimizing J w.r.t. tnj
def maximization():
    for j in range(k):
        ans = np.zeros((d,1))
        count = 0
        for n in range(N):
            ans = ans + int(T[j][n])*(X[:,n].reshape((d,1)))
            count = count + T[j][n]
            
        mu = ans/count
#         print(mu)
        mu_array[:,j] = mu.reshape((2))
#         print(mu_array)
      
def k_means():
    perc = 1
    while(perc > 0.001):
        mu_old = mu_array.copy()
        expectation()
        maximization()
        perc = np.linalg.norm(mu_old - mu_array)/np.linalg.norm(mu_old)
    return mu_array

k_means() 

y_pred = [0] * N
for n in range(N):
     for j in range(k):
        if(T[j,n]==1):
             y_pred[n]= j
                
                    
for k in range(X.shape[1]):
    plt.plot(X[0,k],X[1,k],c[y_pred[k]]+"o")
    
plt.plot(mu_array[0,:],mu_array[1,:],"k"+"s")
    
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Class labels given")
plt.show()   