import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES,FAMOUS_SAMPLES
from pyclustering.utils import read_sample

def read_data(filename):
    """
    Returns data and labels from the dataset
    """

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("mnist.csv")

#segregate data classwise
dig=[]
for i in range(10):
    t=train_data[train_labels==i]
    dig.append(t)

#create an array of colors
col=cm.rainbow(np.linspace(0,1,10))

#Find E and V of origianl data matrix
T=train_data.T
C=np.cov(T)
E,V=np.linalg.eig(C)
E=E.real
V=V.real

D=[]
#Apply PCA to reduce to 2-dimensions
for x in range(10):
    T=dig[x].T
    X=[]
    Y=[]
    for i in range(100):
        t1=np.matmul(T[:,i].T,V[:,0])
        t2=np.matmul(T[:,i].T,V[:,1])
        X.append(t1)
        Y.append(t2)
        D.append([float(t1),float(t2)])
    plt.scatter(X,Y,marker='*',color=col[x])

#plot original data
plt.show()

# Prepare initial centers using K-Means++ method.
initial_centers = kmeans_plusplus_initializer(D, 10).initialize()

# Create instance of K-Means algorithm with prepared centers.
kmeans_instance = kmeans(D, initial_centers)

# Run cluster analysis and obtain results.
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
final_centers = kmeans_instance.get_centers()

# Visualize obtained results
kmeans_visualizer.show_clusters(D, clusters, final_centers)