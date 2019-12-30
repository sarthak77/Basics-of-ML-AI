import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

#import MNIST dataset
from torchvision import datasets


def images_to_vectors(images):
  """
  Convert to vector
  """
  return images.view(images.size(0),784)


def TRAIN(data,labels):
	for i in range(len(NNarr)):
		LOSS[i].append(criterion(NNarr[i](Train_D),Train_L).item())
		optimizers[i].zero_grad()
		output=NNarr[i](data)
		error=criterion(output, labels)
		error.backward()
		optimizers[i].step()


class Net(nn.Module):
  """
  Create a network
  """
  
  def __init__(self,weight_type=None):
    """
    Initialise neurel net
    """
    super(Net,self).__init__()
    self.L1=nn.Sequential(nn.Linear(784,1000),nn.ReLU())
    self.L2=nn.Sequential(nn.Linear(1000,1000),nn.ReLU())
    self.L3=nn.Sequential(nn.Linear(1000,10),nn.Sigmoid())
  

    def init_weights(M):
      """
      Initialise weight distribution
      """
      if(type(M)=="Linear"):
        if(weight_type=="uniform"):
          torch.nn.init.uniform_(M.weight,0,1)
        elif(weight_type=="normal"):
          torch.nn.init.normal_(M.weight,0,1)
        elif(weight_type=="xavier"):
          torch.nn.init.xavier_uniform_(M.weight,0,1)

      
    if(weight_type!=None):
      self.L1.apply(init_weights)
      self.L2.apply(init_weights)
      self.L3.apply(init_weights)

    
  def forward(self,x):
    """
    Apply net
    """
    L=[self.L1,self.L2,self.L3]
    for f in L:
      x=f(x)
    return x


#load data
Train_Data=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor())
# Train_Data=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
print(Train_Data)

#subsample the data
NOS=1000
Train_Data.data=Train_Data.data[:NOS]
Train_Data.targets=Train_Data.targets[:NOS]

#set batch size
B=100
Train_B=torch.utils.data.DataLoader(dataset=Train_Data,batch_size=B,shuffle=True)
Train_D=images_to_vectors(Train_Data.data).float()[:B]
Train_L=Train_Data.targets[:B]

"""
 ------
|Part A|
 ------
"""

#initialise
NOE=200
criterion=nn.CrossEntropyLoss()
NNarr=[Net(),Net(),Net(),Net(),Net(),Net()]
LR=[10,1,0.1,0.01,0.0001,0.00001]
LOSS=[[],[],[],[],[],[]]
optimizers=[]
for i in range(6):
    optimizers.append(torch.optim.SGD(NNarr[i].parameters(),lr=LR[i]))

#train
for epoch in range(NOE):
	for n_batch,(real_batch,labels) in enumerate(Train_B):
		data=images_to_vectors(real_batch)
		TRAIN(data,labels)
		break

#plot
for i in range(6):
	plt.plot(LOSS[i],label=str(LR[i]))
plt.title("Loss function with varing LR")
plt.legend()
plt.show()


"""
 ------
|Part B|
 ------
"""

#initialise
NOE=200
criterion=nn.CrossEntropyLoss()
NNarr=[Net(weight_type="uniform"),Net(weight_type="normal"),Net(weight_type="xavier")]
LR=[1,1,1]
LOSS=[[],[],[]]
optimizers=[]
for i in range(3):
    optimizers.append(torch.optim.SGD(NNarr[i].parameters(),lr=LR[i]))

#train
for epoch in range(NOE):
	for n_batch,(real_batch,labels) in enumerate(Train_B):
		data=images_to_vectors(real_batch)
		TRAIN(data,labels)
		break

#plot
weight_types=["uniform","normal","xavier"]
for i in range(3):
	plt.plot(LOSS[i],label=weight_types[i])
plt.title("Loss function with varing initial_weights")
plt.legend()
plt.show()