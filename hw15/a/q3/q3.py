import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def custom(X,Y):
    """
    Returns custom kernel
    """

    lx=len(X)
    ly=len(Y)
    K=np.zeros([lx,ly])
    for i in range(lx):
      for j in range(ly):
        K[i][j]=(1+X[i][0]*Y[j][0]+Y[j][1]*X[i][1])
        K[i][j]=K[i][j]*K[i][j]
    return K

def tfunc(knum):
    """
    Similar operations for all cases
    """

    svm_obj.fit(D,L)
    XX,YY=np.mgrid[-10:10:200j,-10:10:200j]
    Z=svm_obj.decision_function(np.c_[XX.ravel(),YY.ravel()])
    Z=Z.reshape(XX.shape)
    ax[knum][i].contour(X,Y,Z,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-0.5,0,0.5])
    ax[knum][i].pcolor(X,Y,10*Z,cmap="RdBu")

#initial parameters
D=np.array([[1,1],[-1,-1],[1,-1],[-1,1]])#data matrix
L=[1,1,-1,-1]#label matrix
C=[0.0001,1,10]#checking for different values of c
K=["rbf","poly",custom]#checking for different values of k
knum=0#initial kernel

#initialise X and Y matrix
X=np.linspace(-10,10,200)
X,Y=np.meshgrid(X,X)

#initialise figure
fig = plt.figure(figsize=(10,10))
fig.suptitle('Decision Boundary is in bold and Support Vectors are dotted')
ax=[[fig.add_subplot(331),fig.add_subplot(332),fig.add_subplot(333)],[fig.add_subplot(334),fig.add_subplot(335),fig.add_subplot(336)],[fig.add_subplot(337),fig.add_subplot(338),fig.add_subplot(339)]]

#first kernel
for i in range(len(C)):
    svm_obj=SVC(gamma="auto",C=C[i],kernel=K[knum])
    tfunc(knum)
    ax[knum][i].set_title("RBF kernel with C="+str(C[i]))

#second kernel
knum+=1
for i in range(len(C)):
    svm_obj=SVC(gamma="auto",degree=4,C=C[i],kernel=K[knum])
    tfunc(knum)
    ax[knum][i].set_title("Polynomial(degree=4) kernel with C="+str(C[i]))

#third kernel
knum+=1
for i in range(len(C)):
    svm_obj=SVC(gamma="auto",degree=4,C=C[i],kernel=K[knum])
    tfunc(knum)
    ax[knum][i].set_title("Custom kernel k(x,y)=(1+x.y)^2")

plt.show()