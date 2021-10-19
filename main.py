import sys, os
sys.path.append(os.pardir)

import numpy as np
import DP_net
import matplotlib.pylab as plt
import activation


#from dataset.mnist import load_mnist

net = DP_net.Network("dataset/sample_weight.pkl")
net.forward(np.array([1.0, 0.5]), activation.sigmoid, activation.identity)
