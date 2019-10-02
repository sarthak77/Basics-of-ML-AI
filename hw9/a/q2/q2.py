import matplotlib.pyplot as plt
import numpy as np

def gd(x,eeta):
    cc=[]#convergence criteria after each iter
    error=[]#error after each iter
    val=[]#x after each iter

    noi=100#number of iterations
    for i in range(noi):
        c=-2*x*eeta
        x+=c
        cc.append(c)
        error.append(abs(x-0))
        val.append(x)
    return [cc,error,val]


startpoint=-2
eeta=0.1

cc,error,val=gd(startpoint,eeta)
eeta=1
cc1,error1,val1=gd(startpoint,eeta)
eeta=1.01
cc2,error2,val2=gd(startpoint,eeta)

#print first five values
print("First 5 values for eeta=0.1 and startpoint=-2 are:-")
for i in range(5):
    print("Value after "+str(i+1)+" iteration is "+str(val[i]))

#print minima
print("\nMinima using gradient descent is:")
print(val[99])

#plotting
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(cc,label = "Convergence Criteria vs iteration number for eeta=0.1")
ax1.plot(error,label = "Error vs iteration number for eeta=0.1")
ax1.plot(cc1,label = "Convergence Criteria vs iteration number for eeta=1")
ax1.plot(error1,label = "Error vs iteration number for eeta=1")
ax1.plot(cc2,label = "Convergence Criteria vs iteration number for eeta=1.01")
ax1.plot(error2,label = "Error vs iteration number for eeta=1.01")

ax1.legend()
plt.show()
