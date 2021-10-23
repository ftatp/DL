import numpy as np

###################################################################################
## Oct 23, 2021
## 
## Function collection for calculating error of predicted probabilty and true label
## 
## y : predicted probability
##          Usually a float value between 0~1
## t : truth label
##          A list composed with only 0 and 1 (one-hot encoding)
###################################################################################

def check_length(y, t):
    if len(y) != len(t):
        print("Not abled to calculate mean squared error")
        print("Length of values in parameters are different")
        return false
    return true
 
def mean_squared_error(y, t):
    return np.sum((y - t)**2) / len(y)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # Only the value which is placed in the same index of the truth labeled index in y can effect the whole value

def mean_of_cross_entropy_error_in_batch(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
            
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
