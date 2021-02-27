import numpy as np

input_size = 2 # number of features
lr = 0.1 
epochs = 10

X = np.array([
              [1,2],
              [-1,2],
              [0,-1],
             ])

W = np.array([1.0,-0.8]) # initialize weights

y = np.array([
	      1,
              -1,
	      -1
	      ])

def activation_function(z):
	if z >= 0:
		return 1
	else:
		return -1

for epoch in range(epochs):
	n = 0 # number of correct classifications counter
	print('EPOCH# ' + str(epoch))
	for i in range(y.shape[0]):
		x = X[i]
		z = np.dot(W,x)
		a = activation_function(z)
		error = y[i] - a
		
		if error == 0:
			n = n + 1
		
		delta_w = lr * error * x

		if y[i] != a:
			W = W + delta_w # update the weight vector
		else:
			W = W

	if n == 3:
		print('Congratulations! The perceptron converged successfully.')
		print('Those are the weights:')
		print(W)
		break
