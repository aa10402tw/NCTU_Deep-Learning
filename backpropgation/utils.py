import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


##############
###  Data  ###
##############
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]] )
        distance = (pt[0]-pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
                                          
    plt.show()



#######################
###  Cost Function  ###
#######################
# cost function and its derivative
def MSE_error(y, y_pred):
    error = 1/2 * np.dot((y-y_pred).T, (y-y_pred))
    return np.asscalar(error / y.shape[0])

def derivative_MSE(y, y_pred):
    return -(y-y_pred)/ y.shape[0]

def binary_cross_entropy(predictions, targets, epsilon=1e-12):
    N = predictions.shape[0]
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    bce = -np.sum(targets*np.log(predictions) + (1-targets)*np.log(1-predictions))/N
    return bce

def derivative_BCE(y, y_pred, epsilon=1e-9):
    m = y.shape[0]
    return (y_pred-y) / (m* (y_pred - y_pred*y_pred)+epsilon)

#############################
###  Activation Function  ###
#############################
# Activation function and its derivative 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    s = sigmoid(x)
    return s*(1-s)

def ReLU(x):
    zeros = np.zeros(x.shape)
    return np.maximum(x, zeros)

def derivative_ReLU(x):
    dx = x.copy()
    dx[dx<=0] = 0
    dx[dx>0] = 1
    return dx


# function used to calculate accuracy
def sigmoid_to_label(s):
    pred = s.copy()
    pred[pred>0.5] = 1
    pred[pred<0.5] = 0
    return pred

def acc_ratio(Y, Y_pred):
    diff_mat = Y-Y_pred
    num_diff = np.count_nonzero(diff_mat)
    return 1 - (num_diff) / (Y.shape[0] * Y.shape[1])
