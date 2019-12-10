import numpy as np

def per(D,L):
    maxiter=100
    eeta=1
    W=np.array([1,0,-1])
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
        
        print("Iteration "+str(iter)+":: W = "+str(W))
        iter+=1

        for x in misclass:
            W=W+eeta*x[1]*x[0]

        if(W==prevW).all():
            break
        else:
            prevW=W
        

D=[[1,1,1],[-1,-1,1],[2,2,1],[-2,-2,1],[-1,1,1],[1,-1,1]]
L=[-1,-1,1,-1,1,1]
per(D,L)