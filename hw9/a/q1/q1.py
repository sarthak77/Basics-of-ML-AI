import matplotlib.pyplot as plt
import numpy as np


def lodcoshdiff(w,d,x,y):

	sum=[0,0]
	for i in range(0,len(x)):
        tx=np.array([x[i],1])
		sum += np.tanh(np.dot(w,tx) - y[i]) *(tx)
	return sum/len(x)


def lodcoshloss(w,d,x,y):

	sum=[0,0]
	for i in range(0,len(x)):
        tx=np.array([x[i],1])
		sum += np.log(np.cosh(y[i] - np.dot(w,tx)))
	return sum/len(x)


def huberdiff(w,d,x,y):

    sum=[0,0]
    for i in range(0,len(x)):
        tx=np.array([x[i],1])
        if (abs(y[i] - np.dot(w,tx)) < d):
            sum -= (y[i] - np.dot(w,tx))* tx
        else:
            sum -= d*np.sign(y[i] - np.dot(w,tx))*tx
    return sum/len(x)


def huberloss(w,d,x,y):

    sum=[0,0]
    for i in range(0,len(x)):
        tx=np.array([x[i],1])
        if (abs(y[i] - np.dot(w,tx)) < d):
			sum += 0.5*(y[i]-np.dot(w,tx))*(y[i]-np.dot(w,tx))
        else:
			sum += d*np.abs(y[i]-np.dot(w,tx))-0.5*d*d
    return sum/len(x)


def l1diff(w,d,x,y):

	sum=[0,0]
	for i in range(0,len(x)):
        tx=np.array([x[i],1])
		sum += np.sign(np.dot(w,tx) - y[i]) * tx
	return sum/len(x)


def l1loss(w,d,x,y):

	sum=[0,0]
	for i in range(0,len(x)):
        tx=np.array([x[i],1])
		sum += np.abs(y[i] - np.dot(w,tx))
	return sum/len(x)


#setting data
nop=1000
x = np.linspace(-10,10,nop)
y = x+np.random.normal(0,1,nop)
d = 10
w = np.array([0,0])

#calculating gd
loss = []
WX = []
WY = []

alpha = 0.01
for i in range(0,len(x)):
	w-=alpha*huberdiff(w,d,x,y)
    loss.append(huberloss(w,d,x,y))

	# w-=alpha*logcoshdiff(w,d,x,y)
	# loss.append(logcoshloss(w,d,x,y))

	# w-=alpha*l1diff(w,d,x,y)
	# loss.append(l1loss(w,d,x,y))

	WX.append(w[0])
	WY.append(w[1])
	

#plotting
plt.plot(WX,WY)
plt.title('Gradient Descent using Huber')
# plt.title('Gradient Descent using Logcosh')
# plt.title('Gradient Descent using L1')
plt.show()


plt.scatter(x,y)
lx=[-10,10]
a=np.dot(w,np.array([-10,1]))
b=np.dot(w,np.array([10,1]))
plt.plot(lx,[a,b])
plt.title('Data Huber')
# plt.title('Data Logcosh')
# plt.title('Data L1')
plt.show()


plt.plot(loss)
plt.title('Loss using Huber')
# plt.title('Loss using Logcosh')
# plt.title('Loss using L1')
plt.show()