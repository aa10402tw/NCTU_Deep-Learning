# General Package
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import os

# PyToch Package
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
from torch.utils import data

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TensorInputType = torch.FloatTensor
    TensorLabelType = torch.LongTensor
    
    if verbose:
        pbar_train = tqdm(total=len(train_loader), unit=' batches',  ascii=True)
        pbar_train.set_description("{:^10} ({}/{})".format("Training", 0, num_epochs))
        pbar_val = tqdm(total=len(val_loader), unit=' batches',  ascii=True)
        pbar_val.set_description("{:^10} ({}/{})".format("Validation", 0, num_epochs))
    
    for epoch in range(num_epochs):
        loss_total = 0
        correct, total = 0, 0

        # Training
        net.train()
        if verbose:
            pbar_train.n = 0
            pbar_train.last_print_n = 0 
            pbar_train.set_description("{:^10} ({}/{})".format("Training", epoch+1, num_epochs))
            pbar_train.refresh()
        
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
            
            if verbose:
                pbar_train.set_postfix({
                    'train_loss':
                    '%.4f' % (loss_total / (i+1) ),
                    'train_acc':
                    '%.2f %%' % (correct / total * 100)
                })
                pbar_train.update()
            
        history['train_loss'] += [loss_total / len(train_loader)]
        history['train_acc'] += [correct / total]
            
        loss_total = 0
        correct, total = 0, 0
        
        # Evaluation 
        net.eval()
        if verbose:
            pbar_val.n = 0
            pbar_val.last_print_n = 0
            pbar_val.set_description("{:^10} ({}/{})".format("Validation", epoch+1, num_epochs))
            pbar_val.refresh()
            
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
                
                if verbose:
                    pbar_val.set_postfix({
                        'eval_loss':
                        '%.4f' % (loss_total / (i+1) ),
                        'eval_acc':
                        '%.2f %%' % (correct / total * 100)
                    })
                    pbar_val.update()
        history['val_loss'] += [loss_total / len(val_loader)]
        history['val_acc'] += [correct / total]
        
        #if verbose:
            #pbar_total.update()
    if verbose:
        #pbar_total.close()
        pbar_train.close()
        pbar_val.close()
    return history

def save_net(net, name='Net', only_weight=False):
    if '.' not in name:
        name = name + '.pth'

    if not os.path.exists('weights'):
        os.makedirs('weights')
    path = os.path.join("weights", name)
    if only_weight:
        torch.save(net.state_dict(), path)
    else:
        torch.save(net, path)
        
def load_net(name='Net', only_weight=False, net=None):
    if '.' not in name:
        name = name + '.pth'
    path = os.path.join("weights", name)
    if only_weight:
        net.load_state_dict(torch.load(path))
    else:
        net = torch.load(path)
    return net

def save_history(history, name=''):
    if not os.path.exists('histories'):
        os.makedirs('histories')
    with open('histories/' + name + '.pkl', 'wb') as f:
        pickle.dump(history, f)

def load_history(name=''):
    with open('histories/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def plot_historys(historys, model_names=''):
    plt.figure(figsize=(10,5))
    for history, model_name in zip(historys, model_names):
        plt.plot(history['train_acc'], label='Train(%s)'%model_name)
        plt.plot(history['val_acc'], label='Validation(%s)'%model_name)
        plt.ylim(0.5, 1)
    plt.legend(loc='best')

def compute_mean_std(loader):
    mean = 0.
    std = 0.
    for images, _ in tqdm(train_loader):
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

def plot_confusion_matrix(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    num_instance = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0.0%'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    #cm /= cm_sum
    cm = np.divide(cm,cm_sum)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual Labels'
    cm.columns.name = 'Predicted Labels'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
    plt.show()

def test_network(net, test_loader):
    y_true = []
    y_pred = []
    net.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            print(i, end=',')
            inputs = x_batch.type(TensorInputType).to(device)
            labels = y_batch.type(TensorLabelType).to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(predicted.data.cpu().numpy())
    num_classes = 5
    labels = [i for i in range(num_classes)]
    plot_confusion_matrix(y_true, y_pred, labels)


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