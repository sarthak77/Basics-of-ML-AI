import math
import numpy as np
import matplotlib.pyplot as plt
classA = []
classB = []
w =  [float(x) for x in input().split()]
a,b,c = w
for i in range(50):
    x1 = np.random.uniform(-1,1)
    x2 = np.random.uniform(-1,1)
    classA.append([x1,x2,1])
for i in range(50):
    x1 = np.random.uniform(-1,1)
    x2 = np.random.uniform(-1,1)
    classB.append([x1,x2,1])
classifiedA = []
classifiedB = []
for i in classA:
    if(np.dot(w,i) > 0):
        classifiedA.append(i)
for i in classB:
    if(np.dot(w,i) > 0):
        classifiedA.append(i)

accuracy = (len(classifiedA)+len(classifiedB))/100.0
print("Accuracy : ",accuracy)
x = np.linspace(-1,1,10)
y = (a*x+c)/b
myx = []
myy = []
for i in classA:
    myx.append(i[0])
    myy.append(i[1])
myx = np.array(myx)
myy = np.array(myy)
plt.scatter(myx,myy,c='b')
myx = []
myy = []
for i in classB:
    myx.append(i[0])
    myy.append(i[1])
myx = np.array(myx)
myy = np.array(myy)
plt.scatter(myx,myy,c='r')
plt.plot(x,y,'g')
plt.title(w)
plt.show()