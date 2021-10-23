import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt

from dataset.mnist import load_mnist

import Network
import activationFuncs


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape, t_train.shape) # (60000, 784) , (60000, 10) 784 = 28*28
print(x_test.shape, t_test.shape)   # (10000, 784) , (10000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

#net = Network.Network("dataset/sample_weight.pkl")
#net.forward(np.array([1.0, 0.5]), activation.sigmoid, activation.identity)
