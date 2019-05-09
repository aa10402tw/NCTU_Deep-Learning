import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from dataloader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TensorInputType = torch.FloatTensor
TensorLabelType = torch.LongTensor

def get_activation_function(activation_name):
    if activation_name == 'ELU':
        activation_function = nn.ELU(alpha=1.0) 
    elif activation_name == 'ReLU':
        activation_function = nn.ReLU()
    elif activation_name == 'LeakyReLU':
        activation_function = nn.LeakyReLU()
    else:
        raise Exception('No such activation! %s'%(activation_name))
    return activation_function


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        train_data, train_label, test_data, test_label = read_bci_data()
        if train:
            self.xs = train_data
            self.ys = train_label
        else:
            self.xs = test_data
            self.ys = test_label
        
    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])

def num_of_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct, total

def compute_test_accuracy(net, test_data, test_label):
    inputs = torch.from_numpy(test_data).type(TensorInputType).to(device)
    labels = torch.from_numpy(test_label).type(TensorLabelType).to(device)

    net.eval()
    outputs = net(inputs)
    
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def train_network(net,
                  train_loader,
                  num_epochs,
                  optimizer,
                  criterion,
                  scheduler=None,
                  val_loader=None,
                  verbose=True):

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if verbose:
        pbar = tqdm(total=num_epochs, unit=' epochs', dynamic_ncols=True, ascii=True)

    for epoch in range(num_epochs):
        loss_total = 0
        correct, total = 0, 0

        # Train
        net.train()
        if scheduler is not None:
            scheduler.step()

        for i, (x_batch, y_batch) in enumerate(train_loader):
            inputs = x_batch.type(TensorInputType).to(device)
            labels = y_batch.type(TensorLabelType).to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # record trend
            loss_total += loss.data.cpu().item()
            correct_batch, total_batch = num_of_correct(outputs, labels)
            correct += correct_batch
            total += total_batch
        history['train_loss'] += [loss_total / len(train_loader)]
        history['train_acc'] += [correct / total]

        # Validation
        if val_loader == None:
            if verbose:
                pbar.set_postfix({
                    'train_loss':
                    '%.4f' % (history['train_loss'][-1]),
                    'train_acc':
                    '%.2f' % (history['train_acc'][-1])
                })
                pbar.update()
            continue

        loss_total = 0
        correct, total = 0, 0

        net.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_loader):
                inputs = x_batch.type(TensorInputType).to(device)
                labels = y_batch.type(TensorLabelType).to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # record trend
                loss_total += loss.data.cpu().item()
                correct_batch, total_batch = num_of_correct(outputs, labels)
                correct += correct_batch
                total += total_batch
        history['val_loss'] += [loss_total / len(val_loader)]
        history['val_acc'] += [correct / total]
        if verbose:
            pbar.set_postfix({
                'train_loss': '%.4f' % (history['train_loss'][-1]),
                'train_acc': '%.2f' % (history['train_acc'][-1]),
                'val_loss': '%.4f' % (history['val_loss'][-1]),
                'val_acc': '%.2f' % (history['val_acc'][-1])
            })
            pbar.update()
    if verbose:
        pbar.close()
    return history

def plot_historys(historys, model_names, title=""):
    plt.figure(figsize=(10,5))
    colors = [('salmon', 'red') , ('lightblue', 'blue'), ('lightgreen', 'green')]
    for i, (history, model_name) in enumerate(zip(historys, model_names)):
        plt.plot(history['train_acc'], label='%s_train'%model_name, linestyle='dashed', c=colors[i][0])
        plt.plot(history['val_acc'], label='%s_test'%model_name, c=colors[i][1])
        plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')