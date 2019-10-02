import numpy as np
import matplotlib.pyplot as plt

def getmean(s,k):

    set_data=[]
    for i in range(s):
        set_data.append(np.random.normal(0,1,k))

    mean_of_sets=[]
    for i in set_data:
        mean_of_sets.append(np.mean(i))

    variance=np.cov(mean_of_sets)
    
    return variance

N=1000

#Part-A when s is fixed and k varies
s=N
y1=[]
for k in range(1,N):
    y1.append(getmean(s,k))

#Part-B when k is fixed and s varies
k=N
y2=[]
for s in range(1,N):
    y2.append(getmean(s,k))

#plotting
plt.plot(y1)
plt.xlabel("k")
plt.ylabel("variance")
plt.title("Part-A")
plt.axis()
plt.show()

#plotting
plt.plot(y2)
plt.xlabel("s")
plt.ylabel("variance")
plt.title("Part-B")
plt.show()