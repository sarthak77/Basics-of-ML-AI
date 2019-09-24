import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------part1-------------------------------------
# means
m1=np.arange(1,4,1)
m2=np.arange(4,7,1)

# covariance matrices
c1=np.array([1,0,0,0,2,0,0,0,3]).reshape(3,3)
c2=c1
# --------------------------------end--------------------------------------

# # -------------------------------part2-------------------------------------
# # means
# m1=np.arange(1,4,1)
# m2=np.arange(4,7,1)

# # covariance matrices
# c1=np.array([2,-1,0,-1,2,-1,0,-1,2]).reshape(3,3)
# c2=c1
# # --------------------------------end--------------------------------------

# # -------------------------------part3-------------------------------------
# # means
# m1=np.arange(1,4,1)
# m2=m1

# # covariance matrices
# c1=np.array([2,-1,0,-1,2,-1,0,-1,2]).reshape(3,3)
# c2=np.array([2,-1,1,-1,2,-1,1,-1,2]).reshape(3,3)
# # --------------------------------end--------------------------------------

# data points
mat1 = np.random.multivariate_normal(m1,c1,1000)
mat2 = np.random.multivariate_normal(m2,c2,1000)

# xcor
x1=mat1[:,0]
x2=mat2[:,0]

# ycor
y1=mat1[:,1]
y2=mat2[:,1]

# zcor
z1=mat1[:,2]
z2=mat2[:,2]

# plotting
fig=plt.figure()

# 3d
ax1=fig.add_subplot(221,projection='3d')
ax1.scatter(x1,y1,z1,c='r',marker='x')
ax1.scatter(x2,y2,z2,c='b',marker='o')
ax1.set_title('3d')

# x and y
ax1=fig.add_subplot(222)
ax1.scatter(x1,y1,c='r',marker='x')
ax1.scatter(x2,y2,c='b',marker='o')
ax1.set_title('x and y')

# y and z
ax1=fig.add_subplot(223)
ax1.scatter(y1,z1,c='r',marker='x')
ax1.scatter(y2,z2,c='b',marker='o')
ax1.set_title('y and z')

# z and x
ax1=fig.add_subplot(224)
ax1.scatter(x1,z1,c='r',marker='x')
ax1.scatter(x2,z2,c='b',marker='o')
ax1.set_title('z and x')

plt.show()