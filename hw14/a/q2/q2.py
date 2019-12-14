import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def linreg(W,X,Y):
    """
    Return (y-wTx)^2
    """

    t=np.dot(X,W)
    t=t.T
    t=Y-t[0]
    return np.dot(t.T,t)

def logreg(W,X,Y):
    """
    Return (y-g(wTx))^2
    """

    t=np.dot(X,W)
    t=1/(1+np.exp(-t))
    t=t.T
    t=Y-t[0]
    return np.dot(t.T,t)    



#total samples in each class
N=1000
nos=int(N/2)

#first class
x1=np.random.normal(1,1,nos)
y1=np.ones(nos)

#second class
x2=np.random.normal(2,2,nos)
y2=-y1

#concatanate
X=np.hstack([x1,x2])
X=np.vstack([X,np.ones(N)])#append 1 to each data point
Y=np.hstack([y1,y2])

#total test points
N2=100

#initialise x,y,z1,z2
inp=np.linspace(-10,10,N2)
out1=np.ones([N2,N2])
out2=np.ones([N2,N2])

#create graph
for x in range(N2):
    for y in range(N2):
        W=np.array([[inp[x],inp[y]]])
        out1[x][y]=linreg(W.T,X.T,Y)
        out2[x][y]=logreg(W.T,X.T,Y)

#plotting
fig=plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(inp,inp,out1,200)
ax.contour3D(inp,inp,out2,200)
plt.show()