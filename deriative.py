import numpy as np

def numerical_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2*h)

def numerical_gradient(func , x):
    # x : Can be a vector(numpy array). The length is is number of parameters of func
    #   Must be a vector of float!!! int type will not calculate properly
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i].astype(np.float)
        x[i] = tmp_val + h
        fval2 = func(x)
        x[i] = tmp_val - h
        fval1 = func(x)

        grad[i] = (fval2 - fval1) / (2*h)
        x[i] = tmp_val

    return grad
