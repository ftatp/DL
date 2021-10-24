import numpy as np

def numerical_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2*h)

def numerical_gradient(func, x):
    # x : Can be a vector(numpy array). The length is is number of parameters of func
    #   Must be a vector of float!!! int type will not calculate properly

    shape = x.shape
    #print(shape)
    h = 1e-4
    grad = np.zeros(x.size)
    x = x.reshape(x.size, 1)
    
    for i in range(x.size):
        #print("-->", i)
        tmp_val = x[i]#.astype(np.float)
        #print("x[i] : ", tmp_val)
        #print(x.reshape(shape))
        x[i] = tmp_val + h
        fval2 = func(x.reshape(shape))
        x[i] = tmp_val - h
        fval1 = func(x.reshape(shape))

        grad[i] = (fval2 - fval1) / (2*h)
        x[i] = tmp_val

    grad = grad.reshape(shape)
    return grad

def gradient_descent(func, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(func, x)
        x -= lr * grad

    return x
