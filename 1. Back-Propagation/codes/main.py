import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from tqdm import tqdm

from utils import *
from nn import *

############################
### Train on Linear Data ###
############################
# Hyperparameters
num_epochs = 50000
lr = 0.1
hidden_dims = (4, 4)

# Training on Linear Data
net = NeuralNetwork(hidden_dims=hidden_dims, activation_hidden='ReLU', activation_output='sigmoid', criterion='BCE')
net.summary()
X, Y = generate_linear(n=100)
history = {'loss':[], 'acc':[]}

print("\n\n===== Training on Linear Data =====")
for epoch in range(1, num_epochs+1):
    out = net.forward(X)
    net.backward(Y, out, lr=lr)
    loss = binary_cross_entropy(Y, out)
    Y_pred = sigmoid_to_label(out)
    acc = acc_ratio(Y, Y_pred)
    history['loss'] += [loss]
    history['acc'] += [acc]
    if epoch % (num_epochs//10) == 0:
        print('(epoch %d) loss : %.8f acc : %.2f'%(epoch, loss, acc))


# Plot result
plt.subplot(1,2,1), plt.plot(history['loss']), plt.title('Loss')
plt.subplot(1,2,2), plt.plot(history['acc']), plt.title('Acc'), plt.show()
show_result(X, Y, Y_pred)
print(out[:10])

#########################
### Train on XOR Data ###
#########################
# Hyperparameters
num_epochs = 100000
lr = 0.1
hidden_dims = (10, 10)

# build model and data
net = NeuralNetwork(hidden_dims=hidden_dims, activation_hidden='ReLU', activation_output='sigmoid', criterion='BCE')
net.summary()
X, Y = generate_XOR_easy()
history = {'loss':[], 'acc':[]}

print("\n\n===== Training on XOR Data =====")
for epoch in range(1, num_epochs+1):
    out = net.forward(X)
    net.backward(Y, out, lr=lr)
    loss = binary_cross_entropy(Y, out)
    Y_pred = sigmoid_to_label(out)
    acc = acc_ratio(Y, Y_pred)
    history['loss'] += [loss]
    history['acc'] += [acc]
    if epoch % (num_epochs//10) == 0:
        print('(epoch %d) loss : %.8f acc : %.2f'%(epoch, loss, acc))

# Plot result
plt.subplot(1,2,1), plt.plot(history['loss']), plt.title('Loss')
plt.subplot(1,2,2), plt.plot(history['acc']), plt.title('Acc'), plt.show()
show_result(X, Y, Y_pred)
print(out[:10])