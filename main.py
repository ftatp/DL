import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt

from dataset.mnist import load_mnist

import Network
import activationFuncs
import lossFuncs
import deriative


#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#
#print(x_train.shape, t_train.shape) # (60000, 784) , (60000, 10) 784 = 28*28
#print(x_test.shape, t_test.shape)   # (10000, 784) , (10000, 10)
#
#train_size = x_train.shape[0]
#batch_size = 10
#batch_mask = np.random.choice(train_size, batch_size)
#x_batch = x_train[batch_mask]
#t_batch = t_train[batch_mask]
#
net = Network.Network(activation_func=activationFuncs.sigmoid, output_func=activationFuncs.softmax, loss_func=lossFuncs.cross_entropy_error)
input_layer = np.array([1.0, 0.5])
#output_layer = net.forward(input_layer)
#print(output_layer)

print("################################################")

#np.argmax(output_layer)

t = np.array([0, 1])

#loss = net.loss(input_layer, t)

#print(loss)

def f(W):
    #print("Input layer : ", input_layer)
    return net.loss(input_layer, t)

W = net.network['hidden_layers']
grad = deriative.numerical_gradient(f, W[0])

print(grad)

#def function_2(x):
#    return x[0]**2 + x[1]**2
#
#minVal = deriative.gradient_descent(function_2, np.array([3.0, 4.0]), lr=0.1)
#print(minVal)
#
