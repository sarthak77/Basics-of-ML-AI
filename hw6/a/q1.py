import numpy as np
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

samples=100

N=np.random.normal(10,.1,samples)
x=np.arange(1,samples+1,1)
y=np.sin(x)+N

Mactual=np.mean(y)
Cactual=np.var(y)

X=range(1,int(samples/2)+1,1)

M=[]
C=[]
for k in X:
    Tm=[]
    Tc=[]
    for j in range(0,int(samples/k),1):
        tm=[]
        tc=[]
        for i in range(0,int(samples/k),1):
            if i!=j:
                x=y[i*k:(i+1)*k]
                # print(x)
                tm.append(np.mean(x))
                tc.append(np.var(x))
        Tm.append(abs(Mactual-np.mean(tm)))
        Tc.append(abs(Cactual-np.mean(tc)))
    M.append(np.mean(Tm))
    C.append(np.mean(Tc))

X2=[samples/i for i in X]
# X2=X
# X2.sort()


fig=plt.figure()
plt.plot(X2,M,color='red')
plt.plot(X2,C,color='blue')
# plt.axis('equal')
plt.show()