from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import matplotlib.pyplot as plt

#generate 2 clusters
NOS=20
k=2
X,Y=make_blobs(n_samples=NOS,centers=k,n_features=3,random_state=0)

#calculate error
E=[]
for i in range(2,20,1):
	kmeans=KMeans(n_clusters=i,init='k-means++').fit(X)
	E.append(silhouette_score(X,kmeans.labels_))

#plotting
X=[i for i in range(2,20,1)]
plt.plot(X,E)
plt.xlabel('k')
plt.ylabel('silhouette_score')
plt.show()