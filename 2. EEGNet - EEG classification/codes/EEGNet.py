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

class EEGNet(nn.Module):
    def __init__(self, activation_name='ELU', dropout_ratio=0.5):
        super(EEGNet, self).__init__()
        self.activation_name = activation_name
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1,1), groups=16, bias=False),
            get_activation_function(activation_name),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=dropout_ratio),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32),
            get_activation_function(activation_name),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0), 
            nn.Dropout(p=dropout_ratio),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.depthwise_conv(out)
        out = self.separable_conv(out)
        out = out.view(-1, 736) # flattern
        out = self.classifier(out)
        return out
    
    def get_activation_name(self):
        return self.activation_name

def test_EEGNet():
    net = EEGNet(activation_name='ReLU')
    x = torch.randn((64, 1, 2, 750))
    out = net(x)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    print("# parameters:", params)
    print("\n==== Network Architecture ====")
    print(net)

if __name__ == '__main__':
    test_EEGNet()