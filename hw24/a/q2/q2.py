#importing modules
import numpy as np
import matplotlib.pyplot as plt

#Generate data
N=1000
X1=np.array([[i,i] for i in np.linspace(-100,100,N//2)])
X2=np.array([[i,-i] for i in np.linspace(-100,100,N//2)])
X=np.concatenate([X1,X2])
Y=np.array([1 if i < N//2 else -1 for i in range(N)])

#shuffle data
x=np.arange(N)
np.random.shuffle(x)
y1=list(x[:N//2])
y2=list(x[N//2:N])

#init algo
flag=False
iters=0
max_iters=100

while((not flag) and (iters<max_iters)):
  iters+=1

  X1=X[y1]
  mu1=np.mean(X1,axis=0)
  d1,u1=np.linalg.eig(np.cov(X1.T))
  u1=u1[:,0]

  X2=X[y2]
  mu2=np.mean(X2,axis=0)
  d2,u2=np.linalg.eig(np.cov(X2.T))
  u2=u2[:,0]

  #check shift of points
  for i in range(N):
    x=X[i,:]
    b1=np.linalg.norm(x-mu1)**2-np.dot(u1,x-mu1)**2
    b2=np.linalg.norm(x-mu2)**2-np.dot(u2,x-mu2)**2
    if(b1<=b2) and (i not in y1):
        y1.append(i)
        y2.remove(i)
        flag=False
    elif(b1>b2) and (i not in y2):
        y2.append(i)
        y1.remove(i)
        flag=False

#plotting
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)
ax1.scatter(X[:,0],X[:,1],c=Y)
ax1.set_title("Original Data")
ax2.scatter(X[:,0],X[:,1],c=[1 if i in y1 else 2 for i in range(N)])
ax2.set_title("Converged output")
plt.show()