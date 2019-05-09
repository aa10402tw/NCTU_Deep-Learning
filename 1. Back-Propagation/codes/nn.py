import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

# Network Architecture
class NeuralNetwork: # With batch size
    def __init__(self, in_dims=2, hidden_dims=(10, 10), out_dims=1, 
                 activation_hidden='sigmoid', activation_output='sigmoid',
                 criterion='MSE', with_bias=True):
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        self.num_layers = len(hidden_dims)
        self.W = [None]
        self.W.append(np.random.randn(in_dims, hidden_dims[0])) # W_1
        for l in range(1, len(hidden_dims)):
            self.W.append(np.random.randn(hidden_dims[l-1], hidden_dims[l]))
        self.W.append(np.random.randn(hidden_dims[-1], out_dims)) # W_L
        
        # bias
        self.with_bias = with_bias
        if self.with_bias:
            self.biases = [None]
            for l in range(len(hidden_dims)):
                self.biases.append(np.zeros(hidden_dims[l]) + 0.01)
            self.biases.append(np.zeros(out_dims) + 0.01)

        # activation function for hidden layer 
        if activation_hidden =='ReLU':
            self.activation_hidden = ReLU
            self.derivative_activation_hidden = derivative_ReLU
        elif activation_hidden == 'sigmoid': 
            self.activation_hidden = sigmoid
            self.derivative_activation_hidden = derivative_sigmoid
        else:
            raise Exception("No such activated function [%s] (used in hidden layer) "%(activation_hidden))
            
        # activation function for output layer 
        if activation_output =='ReLU':
            self.activation_output = ReLU
            self.derivative_activation_output = derivative_ReLU
        elif activation_output == 'sigmoid': 
            self.activation_output = sigmoid
            self.derivative_activation_output = derivative_sigmoid
        else:
            raise Exception("No such activated function [%s] (used in output layer) "%(activation_output))
        
        # Cost function 
        if criterion == 'MSE':
            self.criterion = MSE_error
            self.derivative_criterion = derivative_MSE
        elif criterion=='BCE':
            self.criterion = binary_cross_entropy
            self.derivative_criterion = derivative_BCE
        else:
            raise Exception("No such cost function [%s] "%(criterion))
    
    def forward(self, x):
        self.a = [None for l in range(len(self.W))]
        self.z = [None for l in range(len(self.W))]
        self.a[0] = x
        for l in range(1, len(self.W)):
            self.z[l] = np.matmul(self.a[l-1], self.W[l])
            if self.with_bias:
                self.z[l] += self.biases[l]
            if l == len(self.W)-1:
                self.a[l] = self.activation_output(self.z[l])
            else:
                self.a[l] = self.activation_hidden(self.z[l])
        self.out = self.a[-1]
        return self.out

    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, y, out, lr=0.1):
        deltas = [None for l in range(len(self.W))]
        if self.derivative_criterion == derivative_BCE :
            deltas[-1] = out - y
        else:
            deltas[-1] = self.derivative_criterion(y, out) * self.derivative_activation_output(self.z[-1])
        
        for l in range(len(self.W)-2, 0, -1):
            deltas[l] = np.matmul(deltas[l+1], self.W[l+1].T) * self.derivative_activation_hidden(self.z[l]) 
        
        batch_size = self.out.shape[0]
        # Matrix Form     
        for l in range(1, len(self.W)):
            gradient = np.matmul(self.a[l-1].T, deltas[l])
            self.W[l] -= (1/batch_size) * lr * gradient 
            
            if self.with_bias:
                gradient_bias = np.sum(deltas[l], axis=0)
                self.biases[l] -= (1/batch_size) * lr * gradient_bias 
            
    def summary(self):
        print('='*80)
        print("{:^20}  {:^20}  {:^20} ".format('Layer', 'out_dims',  'activation'))
        print('='*80)
        for l in range(1, len(self.W)+1):
            layer_name = 'Hidden_layer_%d' %(l-1) 
            if l==1:
                layer_name = 'Input_layer'
            if l < len(self.W):
                last_dims, next_dims = self.W[l].shape
                activation_name = 'ReLU' if self.activation_hidden == ReLU else 'sigmoid'
            else:
                layer_name = 'Output_layer'
                activation_name = 'ReLU' if self.activation_output == ReLU else 'sigmoid'
                last_dims = next_dims
            print("{:^20} {:^20} {:^20} ".format(layer_name, '(None, %d)'%last_dims, activation_name))
            if l < len(self.W):
                print('-'*80)
        print('='*80)
        cost_function_name = 'Binary Cross Entropy(BCE)' if self.criterion == binary_cross_entropy else 'Mean Square Error(MSE)'
        print("{:^20} {:^40}".format("Cost Function", cost_function_name))
        print('='*80)
        print('\n')
        

def train_network(net, X, Y, num_epochs=10000, lr=0.1, criterion=binary_cross_entropy):
    loss = []
    acc= []
    for epoch in tqdm(range(num_epochs)):
        out = net.forward(X)
        net.backward(Y, out, lr=lr)

        loss.append(criterion(Y, out))

        Y_pred = sigmoid_to_label(out)
        acc.append(acc_ratio(Y, Y_pred))
    return net, loss, acc


