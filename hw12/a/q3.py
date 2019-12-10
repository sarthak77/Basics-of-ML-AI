import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
	data = []
	with open(filename,"r") as f:
		lines = f.readlines()
		for line in lines:
			row = line.split("\t")
			row = np.array(row).astype('float64')
			data.append(row)
	norm_data = []
	mean = np.mean(data, axis=0)

	for row in data:
		temp = []
		for i in range(6):
			temp.append(row[i]/mean[i])
		norm_data.append(temp)
	return norm_data

def MSE_Gradient(w,data):
	neg_grad = np.zeros(6)
	for i in range(len(data)):
		x = data[i]
		x[5] = 1    
		x = np.array(x)
		y = data[5]
		neg_grad = ((np.dot(w,x) - y) * x)
	return neg_grad

def MSE_Loss(w,data):
	loss = 0
	for i in range(len(data)):
		x = data[i]
		x[5] = 1    
		x = np.array(x)
		y = data[i][5]
		loss += ((np.dot(w,x) - y)**2)
	return loss/len(data)

# Normal GDE
def Normal_GDE(data):
	print("Normal GDE")
	alpha = 0.00005
	iters = 0
	w = np.zeros(6)
	prev_loss = 10
	loss = MSE_Loss(w,data)
	while prev_loss > 1e-3 and abs(prev_loss - loss) > 1e-5 :
	# loss is pretty less or loss doesn't change much
		iters += 1
		prev_loss = loss
		print("ITERATION = ",iters,", Loss = ",loss)
		gradient = MSE_Gradient(w,data)
		w = w - alpha*gradient
		loss = MSE_Loss(w,data)

	print("FINAL W = ",w," AFTER ",iters," ITERATIONS ")

# Optimized Learning Rate GDE
def Optimized_Learning_GDE(data):
	print("OPTIMIZED LEARNING RATE GDE")
	iters = 0
	w = np.zeros(6)
	prev_loss = 10
	loss = MSE_Loss(w,data)
	while prev_loss > 1e-3 and abs(prev_loss - loss) > 1e-5 :
	# loss is pretty less or loss doesn't change much
		iters += 1
		prev_loss = loss
		print("ITERATION = ",iters,", Loss = ",loss)
		gradient = MSE_Gradient(w,data)
		hessian = np.zeros([6,6])
		for i in range(0, len(data)):
			x = data[i]
			x[5] = 1
			x = np.array(x)
			hessian += np.outer(x,x)

		alpha_opt = (np.linalg.norm(gradient)**2)/(np.dot(gradient,np.dot(hessian,gradient)))

		w = w - alpha_opt*gradient
		loss = MSE_Loss(w,data)

	print("FINAL W = ",w,"\n AFTER ",iters," ITERATIONS AND LOSS = ",loss)

data = read_data("airfoil_self_noise.dat")
# Normal_GDE(data)
Optimized_Learning_GDE(data)