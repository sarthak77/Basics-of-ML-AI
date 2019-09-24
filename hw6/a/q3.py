# Importing packages
import numpy as np
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parta=[[0,0,2,3,3,2,2],[0,1,0,2,3,2,0]]
partb=[[7,8,9,8,7,8,7],[7,6,7,10,10,9,11]]

# Specifying boundary
boundx=np.arange(0,11,1)
boundy=10-boundx
# boundy=(np.sqrt(-1216*(boundx**2)-4408*boundx+58433) + 30*boundx -29)/46
boundy1 = ((1.92*boundx + 11.54) + ((1.92*boundx+11.54)**2 + 4*0.67*(1.23*boundx**2+38.43*boundx-205.07))**0.5)/(2*0.67)
boundy2 = ((1.92*boundx + 11.54) - ((1.92*boundx+11.54)**2 + 4*0.67*(1.23*boundx**2+38.43*boundx-205.07))**0.5)/(2*0.67)


# Plotting
fig=plt.figure()
plt.scatter(parta[0],parta[1],color='red',marker='x')
plt.scatter(partb[0],partb[1],color='green',marker='o')

plt.plot(boundx,boundy1,color='yellow',linewidth=4)
plt.plot(boundx,boundy2,color='yellow',linewidth=4)

plt.plot(boundx,boundy1-5,color='blue',linewidth=4)
plt.plot(boundx,boundy2+5,color='blue',linewidth=4)

# plt.axis('equal')
plt.show()