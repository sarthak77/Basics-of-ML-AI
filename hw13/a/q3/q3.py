from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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



#read data
train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
# print(train_data.shape, test_data.shape)
# print(train_labels.shape, test_labels.shape)



#preprocessing for LR
D=[]
L=[]
for i in range(10):
    temp=[]
    temp2=[]
    for j in range(len(train_labels)):
        x=train_labels[j]
        if(x==i):
            temp.append(train_data[j])
            temp2.append(i)
    D.append(temp)
    L.append(temp2)
D=np.array(D)
L=np.array(L)



#computing decision boundary(-training-)
wmat=[[LogisticRegression(solver = 'liblinear') for i in range(10)] for j in range(10)]
for i in range(10):
    for j in range(i+1,10,1):
        X=np.vstack((D[i],D[j]))
        Y=np.hstack((L[i],L[j]))
        wmat[i][j].fit(X,Y)


#(-testing-)
count=0
ypred=[]
for i in range(len(test_data)):
    
    x=test_data[i]
    digit=[0,0,0,0,0,0,0,0,0,0]

    for j in range(10):
        for k in range(j+1,10,1):
            digit[int(wmat[j][k].predict(x.reshape(1,-1)))]+=1

    y=digit.index(max(digit))
    ypred.append(y)
    if(y==test_labels[i]):
        count+=1

print('Accuracy is ~ '+str(count/10)+'%')



#plotting of CM
CM=confusion_matrix(test_labels,ypred)
print('Confusion Matrix:')
print(CM)
cmp=plt.matshow(CM)
plt.colorbar(cmp)
plt.show()