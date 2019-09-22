import matplotlib.pyplot as plt
import numpy as np

points=1000

m=np.random.random(size=1)+1
c=np.random.random(size=1)+1

X=np.linspace(0,20,points)
L=[m*x+c for x in X]
Y=[y+2*(np.random.random(size=1)-.5) for y in L]

mx=np.mean(X)
my=np.mean(Y)

X1=[]
for x in X:
    X1.append(x-mx)

Y1=[]
for y in Y:
    Y1.append(y-my)

mat=np.cov(np.array([X,Y]))
print(mat)

eg_val,eg_vec = np.linalg.eig(mat)
# print(eg_vec)

fig=plt.figure()
ax=plt.axes()

ax.plot([mx,mx+15*eg_vec[0][1]],[my,my+15*eg_vec[1][1]],color='r')
ax.plot([mx,mx+15*eg_vec[0][0]],[my,my+15*eg_vec[1][0]],color='g')

plt.scatter(X,Y)
plt.axis('equal')
plt.show()