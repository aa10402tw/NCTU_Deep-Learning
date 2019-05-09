import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# Criterion
def compute_error(y_true, y_pred):
    total = len(y_true)
    num_error = 0
    for y, y_ in zip(y_true, y_pred):
        if y != y_:
            num_error += 1
    return num_error

def isCorrect(y_true, y_pred):
    if np.array_equal(y_true, y_pred):
        return 1
    else:
        return 0

# Plot
def plot_history(history):
    xs = [i*10 for i in range(len(history['bit_error']))]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    # Bit error
    ax1.set_xlabel('# Iterations')
    ax1.set_ylabel('bit error', color=color)
    ax1.plot(xs, history['bit_error'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Accuracy
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, history['accuracy'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


class BinaryAdditionDataGenerator():
    def generate_data(self, n_bits=8):
        x1 = np.random.randint(2, size=n_bits)
        x2 = np.random.randint(2, size=n_bits)
        y  = [0 for i in range(n_bits)]
        carry = 0
        for i, (a, b) in enumerate(zip(x1, x2)):
            y[i] = a + b + carry
            carry = 0
            if y[i] >= 2:
                y[i] %= 2
                carry = 1
        x = np.array([x1, x2]).T
        y = np.array(y)
        return x, y

# Data Conversion
def to_label(out):
    label = []
    for o in out[1:]:
        if o.item() > 0.5:
            label.append(1)
        else:
            label.append(0)
    return np.array(label)

def to_input(x1, x2):
    input = np.array([to_binary_list(x1)[::-1], 
                      to_binary_list(x2)[::-1]]).T
    return input

def to_binary_list(number, n_bits=8):
    bin_str = bin(number)
    bin_str = "0"*n_bits + bin_str[2:]
    bin_list = [int(b) for b in bin_str[-n_bits:]]
    return bin_list

def to_number(bin_array):
    bin_array = np.array(bin_array).astype(str)
    if len(bin_array.shape) == 1:
        bin_str = ''.join(bin_array)
        return int(bin_str, 2)
    else:
        result = []
        for bin_subarray in bin_array.T:
            bin_str = ''.join(bin_subarray)
            result.append(int(bin_str[::-1], 2))
        return result