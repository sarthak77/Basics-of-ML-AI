#import numpy modules
import numpy as np

#import keras modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#import dataset
from keras.datasets import mnist



def applymodel(kernel_initializer):

    #create network
	model=Sequential()
	model.add(Dense(1000,kernel_initializer=kernel_initializer,activation='tanh', input_shape=(784,)))
	model.add(Dense(1000,kernel_initializer=kernel_initializer, activation='tanh'))
	model.add(Dense(2,kernel_initializer=kernel_initializer, activation='sigmoid'))

	model.summary()
	model.compile(loss='MSE',optimizer=RMSprop(),metrics=['accuracy'])
	history = model.fit(X_train, Y_train,batch_size=B,epochs=epochs,verbose=1,validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])



def sep():
    """
    Seperate into classes
    """
    
    X_train,Y_train,X_test,Y_test=[[],[],[],[]]

    for i in range(len(Ytrain)):
    	if Ytrain[i]==0 or Ytrain[i]==1:
    		X_train.append(Xtrain[i])
    		Y_train.append(Ytrain[i])

    for i in range(len(Ytest)):
    	if Ytest[i]==0 or Ytest[i]==1:
    		X_test.append(Xtest[i])
    		Y_test.append(Ytest[i])

    X_train=np.array(X_train)
    X_train=X_train.reshape(X_train.shape[0],784)
    X_train=X_train.astype('float32')
    X_train /= 255

    X_test=np.array(X_test)
    X_test=X_test.reshape(X_test.shape[0],784)
    X_test=X_test.astype('float32')
    X_test /= 255

    Y_train=np.array(Y_train)
    Y_train=keras.utils.to_categorical(Y_train, C)

    Y_test=np.array(Y_test)
    Y_test=keras.utils.to_categorical(Y_test, C)

    return [X_train,X_test,Y_train,Y_test]



#number of classes
C=2

#batch size
B=128

#no of epochs
epochs=20

#load data
(Xtrain,Ytrain),(Xtest,Ytest)=mnist.load_data()

#seperate into classes as mnist has 10 classes
X_train,X_test,Y_train,Y_test=sep()

#apply model with differetn initializers
print("Model with random weights")
applymodel('random_uniform')

print("Model with zero weights")
applymodel('zeros')

print("Model with one weights")
applymodel('ones')