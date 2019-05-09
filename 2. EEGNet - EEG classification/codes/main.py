import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import *
from EEGNet import *
from DeepConvNet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TensorInputType = torch.FloatTensor
TensorLabelType = torch.LongTensor

def plot_historys(historys, model_names, title=""):
    plt.figure(figsize=(10,5))
    colors = [('lightblue', 'blue'), ('salmon', 'red') ,  ('lightgreen', 'green')]
    for i, (history, model_name) in enumerate(zip(historys, model_names)):
        plt.plot(history['train_acc'], label='%s (train)'%model_name, linestyle='dashed', c=colors[i][0])
        plt.plot(history['val_acc'], label='%s (test)'%model_name, c=colors[i][1])
        plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')


def train_EEGNet(activation_name='ReLU', dropout_ratio=0.5):
    net = EEGNet(activation_name=activation_name, dropout_ratio=dropout_ratio)
    net = net.to(device)

    num_epochs = 800
    batch_size = 128
    lr = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 500, 700], gamma=0.25)

    # Training data Loader
    train_dataset = EEGDataset(train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=batch_size, shuffle=True)

    # Test data Loader
    test_dataset = EEGDataset(train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    history = train_network(net, train_loader, num_epochs, optimizer, criterion, scheduler, test_loader)
    return net, history

if __name__ == '__main__':
    dropout_ratio = 0.5
    net, history = train_EEGNet(activation_name='ReLU', dropout_ratio=dropout_ratio)

    historys = [history]
    model_names = ["dropout_"+str(dropout_ratio)]
    title = "EEGNet (batch_size=128, SGD with momentum 0.5)"
    plot_historys(historys, model_names, title=title)
    plt.show()

    train_data, train_label, test_data, test_label = read_bci_data()
    print("Test Accuracy:", compute_test_accuracy(net, test_data, test_label))