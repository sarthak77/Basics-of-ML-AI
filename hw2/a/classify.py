import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


data = []
choice = [-1, 1]

for i in range(100):
	x = np.random.random(2)
	for num in range(len(x)):
		x[num] *= random.choice(choice)
	x = np.append(x,1)	

	data.append(x)

data = np.array(data)
# print(data)

# x1 = np.array([1,1,0])
# x1 = np.array([-1,-1,0])
# x1 = np.array([0,0.5,0])
x1 = np.array([1,-1,5])
# x1 = np.array([1,1,0.3])

A = 0
B = 0
Acc=0

for i in range(50):
	res = np.dot(x1, data[i])
	if res > 0:
		A += 1
	else:
		B += 1

Acc+=A
# print(A, B)

A = 0
B = 0

for i in range(50, 100):
	res = np.dot(x1, data[i])
	if res > 0:
		A += 1
	else:
		B += 1

# print(A, B)
Acc+=B
print(Acc/100)

a, b, c = x1
d = 0

x = np.linspace(-1,1,10)
y = (a*x+c)/b


fig = plt.figure()
plt.title("E")
# fig.patch.set_facecolor('xkcd:mint green')
ax = fig.add_subplot(1,1,1)
ax.scatter(data[0:50,0], data[0:50,1],color='red',marker='X')
ax.scatter(data[50:100,0], data[50:100,1],color='blue',marker='*')

ax = fig.gca()
ax.plot(x,y)

# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z)
plt.show()




	# print(res)

# print(data)