import numpy as np
from utils import *

# Activation Functions
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RNN():
    def __init__(self, in_dims=2, hidden_dims=16, out_dims=1):
        # Model dimensions
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        
        # Model Weights
        self.U = np.random.normal(size=(hidden_dims, in_dims)) * np.sqrt(2/(in_dims+hidden_dims))
        self.V = np.random.normal(size=(out_dims, hidden_dims)) * np.sqrt(2/(hidden_dims+out_dims))
        self.W = np.random.normal(size=(hidden_dims, hidden_dims)) * np.sqrt(2/(hidden_dims+hidden_dims))
        self.b = np.zeros((hidden_dims, 1))
        self.c = np.zeros((out_dims, 1))
        self.zero_grad()
        
    def zero_grad(self):
        # Set Gradients to zeros
        self.gradient_c = np.zeros((self.c.shape))
        self.gradient_b = np.zeros((self.b.shape))
        self.gradient_V = np.zeros((self.V.shape))
        self.gradient_U = np.zeros((self.U.shape))
        self.gradient_W = np.zeros((self.W.shape))
        self.batch_size = 0
        
    def update_weight(self, lr=0.1):
        # Update Model parameters 
        self.c -= lr * (1/self.batch_size) * self.gradient_c
        self.b -= lr * (1/self.batch_size) * self.gradient_b
        self.V -= lr * (1/self.batch_size) * self.gradient_V
        self.U -= lr * (1/self.batch_size) * self.gradient_U
        self.W -= lr * (1/self.batch_size) * self.gradient_W
        self.zero_grad()
    
    def forward(self, xs):
        T = len(xs)
        # Record from t=0 to t=T (t=0 is initial state)
        self.x = [None for t in range(0, T+1)]
        self.a = [None for t in range(0, T+1)]
        self.h = [None for t in range(0, T+1)] 
        self.o = [None for t in range(0, T+1)]
        self.y = [None for t in range(0, T+1)]
        
        # Foward Pass
        self.h[0] = np.zeros((self.hidden_dims, 1))
        for i, x in enumerate(xs):
            t = i+1
            self.x[t] = x.reshape(-1, 1)
            self.a[t] = self.b + np.matmul(self.W, self.h[t-1]) + np.matmul(self.U, self.x[t])
            self.h[t] = tanh(self.a[t])
            self.o[t] = self.c + np.matmul(self.V, self.h[t])
            self.y[t] = sigmoid(self.o[t])
        return self.y
    
    def backward(self, y_true, y_pred):
        T = len(self.h) - 1 
        
        # Compute dL/do
        d_o = [None for t in range(0, T+1)]
        for t in range(1, T+1):
            i = t - 1
            d_o[t] = y_pred[t] - y_true[i]
        
        # Compute H^(t)
        H = [None for t in range(0, T+1)]
        for t in range(1, T+1):
            H[t] = np.zeros((self.hidden_dims, self.hidden_dims))
            for d in range(H[t].shape[0]):
                H[t][d, d] = 1-(self.h[t][d]**2)
                
        # Compute dL/dh
        d_h = [None for t in range(0, T+1)]
        d_h[-1] = np.matmul(self.V.T, d_o[-1])
        for t in range(T-1, 0, -1):
            d_h[t] = np.matmul(self.W.T, np.matmul(H[t+1], d_h[t+1])) + np.matmul(self.V.T, d_o[t])
        
        # Compute gradient with respect to model parametes {c, b, V, U, W}
        for t in range(1, T+1):
            self.gradient_c += d_o[t]
            self.gradient_b += np.matmul(H[t], d_h[t])
            self.gradient_V += np.matmul(d_o[t], self.h[t].T)
            self.gradient_U += np.matmul(np.matmul(H[t], d_h[t]), self.x[t].T)
            self.gradient_W += np.matmul(np.matmul(H[t], d_h[t]), self.h[t-1].T)    
        self.batch_size += 1
        
    def predict(self, xs):
        out = self.forward(xs)
        out = np.array([o.item() for o in out[1:]])
        out[out>=0.5] = 1
        out[out<0.5] = 0
        return out.astype(np.int32)
    
    def predict_number(self, x1, x2):
        input = to_input(x1, x2)
        output = self.predict(input)
        prediction = to_number(output[::-1])
        return prediction