import numpy as np
import DP_net
import matplotlib.pylab as plt
import activation

net = DP_net.Network()
net.forward(np.array([1.0, 0.5]), activation.sigmoid, activation.identity)
