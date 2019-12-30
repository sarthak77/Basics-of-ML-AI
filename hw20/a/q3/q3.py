#Import modules
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

import keras
from keras import regularizers
import keras.optimizers
import keras.initializers

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Dense, Dropout
from keras.layers import Input, add

from keras.datasets import mnist


"""
Load the data
NOTrS:training samples
NOTeS:test samples
TRC:training class
TRL:training labels
TEC:test class
TEL:test labels
"""

NOTrS=10000
NOTeS=1000
(TRC,TRL),(TEC,TEL)=mnist.load_data()

TRC=TRC.reshape(len(TRC),28*28)
TRC=TRC[0:NOTrS]
TRL=TRL[0:NOTrS]
TRC=TRC/255
predtest=keras.utils.to_categorical(TEL,10)

TEC=TEC.reshape(len(TEC),28*28)
TEC=TEC[0:NOTeS]
TEL=TEL[0:NOTeS]
TEC=TEC/255
predtrain=keras.utils.to_categorical(TRL,10)


#Initialise parameters
NOE=20
B=128
temp=[10,50,100,300,400,500]
encoders=[]


#Allpy NN
for i in range(6):
    x=Input(shape=(784,))
    H1=Dense(temp[i],activation='relu')(x)
    h=Dense(temp[i]//2,activation='relu')(H1)
    H2=Dense(temp[i],activation='relu')(h)
    r=Dense(784,activation='sigmoid')(H2)
    autoencoder=Model(inputs=x,outputs=r)
    autoencoder.compile(optimizer=keras.optimizers.Adam(),loss='mse')
    history=autoencoder.fit(TRC,TRC, batch_size=B, epochs=NOE, verbose=0, validation_data=(TEC,TEC))
    encoders.append(Model(autoencoder.input,autoencoder.layers[-3].output))


#Raw model
c=.1
MR=SVC(C=c,kernel='rbf')
MR.fit(TRC,TRL)
raw_pred=MR.predict(TEC)


#Find accuracy from raw model
ACCR=0
for i in range(len(TEL)):
    if(raw_pred[i]==TEL[i]):
        ACCR=ACCR+1


#Find accuracy from auto E
ACC=[]
model_encode=SVC(C=c,kernel='rbf')

for i in range(6):
    E=encoders[i]
    entr=E.predict(TRC)
    ente=E.predict(TEC)
    model_encode.fit(entr,TRL)
    out=model_encode.predict(ente)
    
    ACCEN=0
    for i in range(len(TEL)):
        if(out[i]==TEL[i]):
            ACCEN=ACCEN+1
    ACC.append(ACCEN/10)


#plotting

#calculate X and Y
Y=[temp[i]+temp[i]//2 for i in range(len(temp))]
Y.append("Raw")
X=np.arange(7)
ACC.append(ACCR/10)

plt.bar(X,ACC, align='center', alpha=0.5)
plt.ylabel('Accuracy')
plt.xticks(X, Y)
plt.title('SVM classifier with RBF kernel')
plt.tight_layout()
plt.show()