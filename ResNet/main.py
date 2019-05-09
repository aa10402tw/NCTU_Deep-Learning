# General Package
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
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

from utils import *
from resnet import *

# Dataset 
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyDataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        self.img_names, self.labels = getData(mode)
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index] + ".jpeg"
        label = self.labels[index]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        if self.transform!= None:
            img = self.transform(img)
        return img, label

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TensorInputType = torch.FloatTensor
    TensorLabelType = torch.LongTensor

    # Transform
    img_size = (512,512)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize((0.3749, 0.2602, 0.1857), (0.2526, 0.1780, 0.1291)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.3749, 0.2602, 0.1857), (0.2526, 0.1780, 0.1291)),
    ])

    # Train resnet50 
    # Init network 
    model_name = 'resnet50(pretrained)'
    net = pretrained_resnet50() 
    net = net.to(device)

    optimizer_ft = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = None
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    batch_size = 16

    # Datasets
    data_root = "data"
    train_dataset = RetinopathyDataset(data_root, "train", train_transform)
    test_dataset = RetinopathyDataset(data_root, "test", test_transform)

    # Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training
    history = train_network(net, train_loader, num_epochs, optimizer_ft, criterion, scheduler, test_loader)

    # Save weights and history
    save_net(net, name=model_name)
    save_history(history, name=model_name)