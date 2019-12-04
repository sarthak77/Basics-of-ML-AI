# importing modules
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


# function for reading data
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]

    return (data, labels)


# function for calculating differentiation
def diff(Z,X):
    A=X@Z@Z.T-X
    C=np.linalg.norm(A)
    Ans=(X.T@A@Z)/C-(A.T@X@Z)/C
    return Ans
    
    
#read data
train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
train_data=train_data-np.mean(train_data,axis=0)
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)


#normal PCA
T=train_data.T
C=np.cov(T)
E,V=np.linalg.eig(C)
E=E.real
V=V.real
X=[]
Y=[]
for i in range(6000):
    X.append(np.matmul(T[:,i].T,V[:,0]))
    Y.append(np.matmul(T[:,i].T,V[:,1]))
plt.scatter(X,Y,marker='*')
plt.title("Normal PCA")
plt.show()


#PCA using GD
W=np.random.rand(T.shape[0],2)
W,_,_=np.linalg.svd(W)
W=W[:,:2]

eeta=1e-5
iterno=100
while iterno>0:
    print(iterno)
    iterno-=1
    W+=eeta*diff(W,train_data)

X=[]
Y=[]
for i in range(6000):
    X.append(-np.matmul(T[:,i].T,W[:,0]))
    Y.append(-np.matmul(T[:,i].T,W[:,1]))
plt.scatter(X,Y,marker='*')
plt.title("PCA using GD")
plt.show()