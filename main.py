import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt

from dataset.mnist import load_mnist

import Network
import activationFuncs


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

#net = Network.Network("dataset/sample_weight.pkl")
#net.forward(np.array([1.0, 0.5]), activation.sigmoid, activation.identity)
