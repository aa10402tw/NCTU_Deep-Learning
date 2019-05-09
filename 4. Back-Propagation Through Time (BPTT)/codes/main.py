import numpy as np

from utils import *
from rnn import *

def train(net, data_generator, batch_size=8, lr=0.05, n_iters=10000, print_every=1000, verbose=True):
    error_total = 0
    num_correct = 0
    history = {'bit_error':[], 'accuracy':[]}
    for iter_ in tqdm(range(1, n_iters+1),  ascii=True):
        # Batch Training
        for batch in range(batch_size):
            input, label = data_generator.generate_data()
            y_pred = net.forward(input)
            net.backward(label, y_pred)
            predict = net.predict(input)
            error_total += compute_error(label, predict) # Bits error
            num_correct += isCorrect(label, predict)     # output match y
        net.update_weight(lr=lr)
        # Print and record history
        if iter_ % print_every == 0:
            bit_error = (error_total / batch_size) / print_every
            accuracy = (num_correct / batch_size) / print_every
            if iter_%1000==0:
                print("(Iter %05d) Bit error: %.3f, Accuracy: %.2f%%"%(iter_, bit_error, accuracy*100))
            history['bit_error'] += [bit_error]
            history['accuracy'] += [accuracy]
            error_total = 0
            num_correct = 0
    return history


# Create Model
net = RNN(in_dims=2, hidden_dims=16, out_dims=1)

# Create Data Generator
data_generator = BinaryAdditionDataGenerator()

# Training parameters
batch_size = 8
lr = 0.005
n_iters = 10000
print_every = 10

# Training
history = train(net, data_generator, batch_size=batch_size, lr=lr, n_iters=n_iters, 
                print_every=print_every, verbose=False)

plot_history(history)


import random 
x1 = random.randint(0, 127)
x2 = random.randint(0, 127)
y_pred = net.predict_number(x1, x2)
print("input : (%d, %d)"%(x1, x2))
print("prediction : %d "%(y_pred))