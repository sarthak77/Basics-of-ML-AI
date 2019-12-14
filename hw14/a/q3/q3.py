#import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA



def read_data(filename):
    """
    Function to read data
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



def parta():
    """ 
    Print accuracy after training model using SVM
    """

    #training
    Model=SVC(kernel='linear',C=1)
    Model.fit(traindata,trainlabel)

    #testing
    acc=0
    pred=[]

    for i in range(len(testdata)):
        x=testdata[i]
        y=testlabel[i]

        ypred=Model.predict(x.reshape(1,-1))
        pred.append(ypred)
        if(ypred==y):
            acc+=1
    
    print("Accuracy using SVM is:- "+str(int(acc/2))+"%")

    return Model



def reduce_dim(D):
    """
    Apply PCA to reduce data to 2 dimensions
    """

    pca1=PCA(n_components=2)
    pca1.fit(D)
    D=pca1.transform(D)
    U=pca1.components_
    
    return [D,U]



def plot_line():
    """
    Draws decision boundary given by SVM
    """

    W=Model.coef_
    W=np.array(W)
    W=np.dot(U,W.T)#reduce
    
    X=np.linspace(-500,500,2)
    C=Model.intercept_
    Y=-(W[0]*X+C)/(W[1])

    return [X,Y]



if __name__ == "__main__":

    train_data, train_labels = read_data("sample_train.csv")
    test_data, test_labels = read_data("sample_test.csv")

    #specify classes
    C1=1
    C2=2

    #splitting data for class 1
    traindata1=train_data[train_labels==C1]
    trainlabel1=train_labels[train_labels==C1]
    testdata1=test_data[test_labels==C1]
    testlabel1=test_labels[test_labels==C1]

    #splitting data for class 2
    traindata2=train_data[train_labels==C2]
    trainlabel2=train_labels[train_labels==C2]
    testdata2=test_data[test_labels==C2]
    testlabel2=test_labels[test_labels==C2]

    #concatanate
    traindata=np.vstack([traindata1,traindata2])
    trainlabel=np.hstack([trainlabel1,trainlabel2])
    testdata=np.vstack([testdata1,testdata2])
    testlabel=np.hstack([testlabel1,testlabel2])

    #calculate model
    Model=parta()
    #reduce traindata
    traindata,U=reduce_dim(traindata)
    #calculate support vectors
    supvec=Model.support_vectors_
    supvec,U1=reduce_dim(supvec)



    #plotting
    #1)support vectors
    for i in supvec:
        plt.scatter(i[0],i[1],color='r')

    #2)class points
    for i in range(len(traindata)):
        p=traindata[i]
        if(trainlabel[i]==C1):
            plt.scatter(p[0],p[1],color='b')
        else:
            plt.scatter(p[0],p[1],color='g')

    #3)line
    X,Y=plot_line()
    plt.plot(X,Y,'-r')
    
    plt.show()