# Importing packages
import numpy as np
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NOS=5000

# For first class
M1=[3,3]
S1=[[3,0],[0,3]]
# S1=[[3,1],[2,3]]
parta=[[],[]]
X1,Y1=np.random.multivariate_normal(M1,S1,NOS).T
for i in range(NOS):
    if((X1[i]<=10 and X1[i]>=0) and (Y1[i]<=10 and Y1[i]>=0) and (len(parta[0])<1000)):
        parta[0].append(X1[i])
        parta[1].append(Y1[i])


# For second class
M2=[7,7]
S2=[[3,0],[0,3]]
# S2=[[7,2],[1,7]]
partb=[[],[]]
X2,Y2=np.random.multivariate_normal(M2,S2,NOS).T
for i in range(NOS):
    if((X2[i]<=10 and X2[i]>=0) and (Y2[i]<=10 and Y2[i]>=0) and (len(partb[0])<1000)):
        partb[0].append(X2[i])
        partb[1].append(Y2[i])


# Specifying boundary
boundx=np.arange(0,11,1)
boundy=10-boundx
# boundy=(np.sqrt(-1216*(boundx**2)-4408*boundx+58433) + 30*boundx -29)/46


# Plotting
fig=plt.figure()
plt.scatter(parta[0],parta[1],color='red',marker='x')
plt.scatter(partb[0],partb[1],color='green',marker='o')
plt.plot(boundx,boundy,color='yellow',linewidth=4)
plt.axis('equal')
plt.show()