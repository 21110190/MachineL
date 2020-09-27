# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:32:14 2020

@author: annik
"""

from scipy.stats import multivariate_normal as norm
from numpy import linalg as LA
import math 

d,N = X.shape
k=2

pi_array = np.zeros(k)

mu = np.zeros((d,k))
cov_matrix = np.zeros((k,d,d))
col_cov = np.array([[[2.0, 0.3], [0.3, 0.5]],[[2.0, 0.3], [0.3, 0.5]]])
mu = np.array([[-1.94876553,  2.0212828 ],[-1.9852701 ,  1.99647112]])

r = np.zeros((N,k))



def initialize_coef():
    # initialize mean
    k_means()
#     print(mu_array)
#initialize mixture coefficient
    for j in range(k):
        Nj = 0
        for n in range(N):
            if(y_pred[n]==j):
                Nj = Nj + 1
#         print(Nj)
        pi_j = Nj/N
#         print(pi_j)
        pi_array[j] = pi_j   
#     print(pi_array)
               
    cov_matrix = np.ones((k,d,d))
#     print(cov_matrix)
#     print(col_cov)
 

    
def expect():
    k=2
    
    gamma_znj = 0.
    for n in range(N): 
        den_sum = 0
        for i in range(k):
            den = pi_array[i]*norm.pdf(X[:,n], mu_array[:,i], col_cov[i])
            den_sum = den_sum + den 
        for j in range(k): 
            num = pi_array[j]*norm.pdf(X[:,n], mu_array[:,j], col_cov[j]) 
            r[n][j] = num/den_sum
#     print(r)
        
    
def maxi():
    k=2
    for j in range(k):
        sum_mean = np.zeros((1,d))
        sum_cov = np.zeros((d,d))
        Nj = 0
        for n in range(N):
            sum_mean = sum_mean + r[n][j]*X[:,n]
            
            minus = X[:,n].reshape((2,1)) - mu_array[:,j].reshape((2,1)) 
            
            sum_cov = sum_cov + r[n][j]*( np.dot( minus, minus.T ) )
            
            Nj = Nj + r[n][j]
            
        pi_array[j] = Nj/N
    
        mu_array[:,j] = (1/(Nj))*sum_mean
        
        col_cov[j] =  (1/(Nj))*sum_cov
        
#     print(pi_array)

    
initialize_coef()
perc = 1


while(perc > 0.05):
    col_cov_old = col_cov.copy()
    expect()
    maxi()
    perc = (np.linalg.norm( (col_cov_old - col_cov)/ (col_cov_old))) 
    
# print(mu_array)
# print(col_cov)  


# y_pred = [0] * N
# for n in range(N):
#      for j in range(k):
#         if(T[j,n]==1):
#              y_pred[n]= j
    
# for k in range(X.shape[1]):
#     plt.plot(X[0,k],X[1,k],'ro', alpha=r[k][0]/4)

# for k in range(X.shape[1]):
#     plt.plot(X[0,k],X[1,k],'go', alpha=r[k][1]/4)
   

    
# for k in range(X.shape[1]):
#     plt.plot(X[0,k],X[1,k],c[y_pred[k]]+"o")

# print(col_cov[0])
nstd = 2
w0, v0 = LA.eig(col_cov[0])
print(v0)
print(w0)
w1, v1 = LA.eig(col_cov[1])
var = col_cov[0,0,0]
var1 = col_cov[0,1,1]
width0 = 2*math.sqrt(var)*nstd 
height0 = 2*math.sqrt(var1)*nstd
width1 = 2*math.sqrt(w1[0])*nstd 
height1 = 2*math.sqrt(w1[1])*nstd 
# x0 = v0[0,:]
# y0 = np.linalg.inv(x0) 
# vec = np.dot(x0,y0)
angle0 = math.atan2(v0[1,0], v0[0,0]) 
angle1 = math.atan2(v1[1,0], v1[0,0]) 


if(angle0 < 0):
      angle0 += 6.28318530718;

#         Conver to degrees instead of radians
angle0 = 180*angle0/3.14159265359;
# print(angle0)

if(angle1 < 0):
      angle1 += 6.28318530718;

#         Conver to degrees instead of radians
angle1 = 180*angle1/3.14159265359;
# print(angle1)

plt.figure()
ax = plt.gca()


ellipse0 = Ellipse(xy=(mu_array[0,0], mu_array[1,0]), width=width0, height=height0, angle = angle0,
                        edgecolor='r', fc='None')
ellipse1 = Ellipse(xy=(mu_array[0,1], mu_array[1,1]), width=width1, height=height1, angle = angle1,
                        edgecolor='g', fc='None')
ax.add_patch(ellipse0)
ax.add_patch(ellipse1)

for n in range(N):
    plt.plot(X[0,n],X[1,n],'o',color=(r[n][0],r[n][1],0,0.1))

plt.axis('tight')
     
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Class labels given")
plt.show() 