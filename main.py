import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt

from dataset.mnist import load_mnist

import network
import activationFuncs
import lossFuncs
import deriative


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#print(x_train.shape, t_train.shape) # (60000, 784) , (60000, 10) 784 = 28*28
#print(x_test.shape, t_test.shape)   # (10000, 784) , (10000, 10)

train_size = x_train.shape[0]
batch_size = 100

net = network.Network(input_size=784, output_size=10, num_of_hidden_layers=1)

for i in range(net.depth):
    print(net.network['weights'][i].shape)
    print(net.network['bias'][i].shape)

net.set_network_training_methods(activation_func=activationFuncs.sigmoid, output_func=activationFuncs.softmax, loss_func=lossFuncs.cross_entropy_error)

iter_num = 10000
train_loss_list = []

learning_rate = 0.1
for i in range(iter_num):
    batch_mask = np.random.choice(train_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = net.numerical_gradient(x_batch, t_batch)

    for i in range(net.depth):
        net.network['weights'][i] -= learning_rate * grads['weights'][i]
        net.network['bias'][i] -= learning_rate * grads['bias'][i]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print("loss : ", loss)
    
