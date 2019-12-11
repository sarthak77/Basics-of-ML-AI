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



#training
Models = LogisticRegression(solver = 'lbfgs',multi_class = 'auto',max_iter = 10000).fit(train_data,train_labels)



#testing
count=0
ypred=[]
for i in range(len(test_data)):
	y=Models.predict(test_data[i].reshape(1,-1))
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