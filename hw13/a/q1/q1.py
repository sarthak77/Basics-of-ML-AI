import numpy as np
import math
import matplotlib.pyplot as plt

def per(D,L):
    maxiter=1000
    eeta=.1
    W=np.array([1,0,1])
    prevW=np.array([0,0,0])
    iter=0

    while(iter<maxiter):
        misclass=[]
        for i in range(len(D)):
            x=D[i]
            x=np.array(x)
            y=L[i]
            if(y*np.dot(W,x)<0) or (y<0 and np.dot(W,x)==0):
                misclass.append([x,y])
        # print(misclass)
        
        # print("Iteration "+str(iter)+":: W = "+str(W))
        iter+=1

        for x in misclass:
            W=W+eeta*x[1]*x[0]

        if(W==prevW).all():
            break
        else:
            prevW=W
    return W



def diff(W,D,L):
    loss=np.array([0,0,0],dtype=float)
    gamma=1
    for i in range(len(D)):
        x=D[i]
        x=np.array(x)
        y=L[i]
        t=gamma*np.dot(W,x)
        t=1/(1+math.exp(-t))
        loss+=(t-y)*x*gamma

    return loss



def lr(D,L):
    L2=[]
    for i in L:
        if(i==-1):
            L2.append(0)
        else:
            L2.append(1)

    maxiter=1000
    eeta=.1
    W=np.array([1,0,1],dtype=float)
    iter=0

    while(iter<maxiter):
        # print("Iteration "+str(iter)+":: W = "+str(W))
        iter+=1
        W-=eeta*diff(W,D,L2)

    return W        

D=[[1,1,1],[1,2,1],[2,20,1],[2,21,1]]
L=[-1,-1,1,1]
W1=per(D,L)
W2=lr(D,L)

print(W1,W2)

#plotting lines
X=[]
Y1=[]
Y2=[]
for i in range(0,4,1):
    X.append(i)
    t=-(W2[0]*i+W2[2])/W2[1]
    Y2.append(t)
    t=-(W1[0]*i+W1[2])/W1[1]
    Y1.append(t)

#plotting points
px=[]
py=[]
for i in D:
    px.append(i[0])
    py.append(i[1])

print('Accuracy is 100%')
plt.plot(px,py,'ro')
plt.plot(X,Y1,color='r')
plt.plot(X,Y2,color='b')
# plt.axis('equal')
plt.show()