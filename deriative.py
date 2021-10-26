import numpy as np

def numerical_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2*h)

def numerical_gradient(func, x):
    # x : Can be a vector(numpy array). The length is is number of parameters of func
    #   Must be a vector of float!!! int type will not calculate properly

    shape = x.shape
    #print(shape)
    print("x : \n", x)
    y = func(x)
    print("y : \n", y)
    h = 1e-4
    grad = np.zeros(x.size)
    x = x.flatten()
    for i in range(x.size):
        print("x's element :", i)
        tmp_val = x[i]#.astype(np.float)
        print("temp ele value : ", tmp_val)
        x[i] = tmp_val + h
        print("temp ele value : ", x[i])
        print("tmp x : \n", x.reshape(shape))
        fval2 = func(x.reshape(shape))
        print("tmp y : \n", fval2)
        x[i] = tmp_val - h
        print("temp ele value : ", x[i])
        print("tmp x : \n", x.reshape(shape))
        fval1 = func(x.reshape(shape))
        print("tmp y : \n", fval1)

        grad[i] = (fval2 - fval1) / (2*h)
        x[i] = tmp_val
        print("\n")

    grad = grad.reshape(shape)
    return grad

def gradient_descent(func, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(func, x)
        x -= lr * grad

    return x
