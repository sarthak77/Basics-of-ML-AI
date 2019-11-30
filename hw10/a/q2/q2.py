import numpy as np
import matplotlib.pyplot as plt

#read data
with open('wine.data') as f:
    lines=f.readlines()
for i in range(len(lines)):
    lines[i]=lines[i].strip()
    lines[i]=lines[i].split(',')

#convert to float
for i in range(len(lines)):
    for j in range(len(lines[i])):
        lines[i][j]=float(lines[i][j])

#separate categories
lines=np.array(lines)
category=lines[:,0]
lines=lines[:,1:]
lines=lines.T

#calculate E,V
C=np.cov(lines)
E,V=np.linalg.eig(C)
#sort E
I=E.argsort()[::-1]
E=E[I]
V=V[:,I]

##############A-PART################
# plt.plot(E)
# plt.show()
##############A-PART################

##############B-PART################
SE=np.sum(E)#sum of E
temp=0
NE=0#no of E whose sum > P
P=.95#% of data

for i in E:
    temp+=i
    NE+=1
    if(temp/SE>P):
        break
# print(NE)
##############B-PART################

##############C-PART################
V2=V[:,0:2]#select first 2 EV
Y=np.matmul(lines.T,V2)#project

c1i,c2i,c3i=[0,0,0]#class indexes
category=list(category)

#find no of items in each class
c1i=len(category)-1-category[::-1].index(1) 
c2i=len(category)-1-category[::-1].index(2) 
c3i=len(category)-1-category[::-1].index(3) 
# print(c1i,c2i,c3i)

#plotting
plt.scatter(Y[0:c1i,0],Y[0:c1i,1],c='b',marker='$1$')
plt.scatter(Y[c1i+1:c2i,0],Y[c1i+1:c2i,1],c='r',marker='$2$')
plt.scatter(Y[c2i+1:c3i,0],Y[c2i+1:c3i,1],c='g',marker='$3$')
plt.axis('equal')
plt.show()
##############C-PART################