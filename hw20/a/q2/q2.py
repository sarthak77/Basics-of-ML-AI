#Import modules
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import keras
import keras.optimizers
import keras.initializers

from keras import regularizers

from keras.models import Sequential
from keras.models import Model

from keras.layers import Dense, Dropout
from keras.layers import Input, add
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape

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

# NOTrS=
# NOTeS=
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


#Initialise Parameters
NOE=20
NOO=4
B=128
LOSS=[]
H=[10,50,100,300,400,500]


#Initialise optimizers
opt1=keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False)
opt2=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt3=keras.optimizers.Adam()
opt4=keras.optimizers.RMSprop()
optimizers=[opt1,opt2,opt3,opt4]


#Apply NN
for j in range(NOO):
    for i in range(len(H)):
        x=Input(shape=(784,))
        H1=Dense(H[i],activation='relu')(x)
        h=Dense(H[i]//2,activation='relu')(H1)
        H2=Dense(H[i],activation='relu')(h)
        r=Dense(784,activation='sigmoid')(H2)
        autoencoder=Model(inputs=x,outputs=r)
        autoencoder.compile(optimizer=optimizers[j],loss='mse')
        pred=autoencoder.predict(TEC)
        LOSS.append(mean_squared_error(TEC,pred))


#Calculate loss
L=[]
for j in range(4):
    temp=[]
    for i in range(6):
        temp.append(LOSS[j*6+i])
    L.append(temp)

#Plotting
N=[H[i]+H[i]//2 for i in range(len(H))]
T=["SGD without momentum","SGD with Momentum","Adam","RMSprop"]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(N,L[i])
    plt.xlabel('Number Of Neurons in hidden layers')
    plt.ylabel('Reconstruction Loss')
    plt.title(T[i])
plt.show()