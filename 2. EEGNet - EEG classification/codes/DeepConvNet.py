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

class DeepConvNet(nn.Module):
    def __init__(self, activation_name='ELU',dropout_ratio=0.5):
        super(DeepConvNet, self).__init__()
        self.activation_name = activation_name
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=True),
            nn.BatchNorm2d(25),
            get_activation_function(activation_name),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=dropout_ratio),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(50),
            get_activation_function(activation_name),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=dropout_ratio),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(100),
            get_activation_function(activation_name),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=dropout_ratio),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(200),
            get_activation_function(activation_name),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
            nn.Dropout(p=dropout_ratio),
        )
        self.fc = nn.Linear(8600, 2)
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out
    
    def get_activation_name(self):
        return self.activation_name

def test_DeepConvNet():
    net = DeepConvNet(activation_name='ReLU')
    x = torch.randn((1, 1, 2, 750))
    out = net(x)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    print("# parameters:", params)
    print("\n==== Network Architecture ====")
    print(net)

if __name__ == '__main__':
    test_DeepConvNet()