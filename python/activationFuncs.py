import numpy as np

def stair(x):
    y = x > 0
    return np.array(y, np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def softmax(x):
    #c = np.max(x)
    #exp_x = np.exp(x - c)
    #sum_exp_x = np.sum(exp_x)
    #return exp_x / sum_exp_x
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
