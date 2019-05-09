from utils import *

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

def conv_3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

def conv_1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def downsample_1x1(in_channels, out_channels, stride=1):
    downsample = nn.Sequential(
        conv_1x1(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels),
    )
    return downsample

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # first conv_3x3 (if stride > 1, do stride in first conv_3x3)
        self.conv1 = conv_3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # second conv_3x3
        self.conv2 = conv_3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If have stride or in_channels does not match with out_channels, identity have to do subsampling
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = downsample_1x1(in_channels, out_channels, stride)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, reduced_channels, stride=1, groups=1):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Use conv_1x1 to reduce # channels first
        self.conv1 = conv_1x1(in_channels, reduced_channels)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        
        # Do conv_3x3 with reduced_channels (if stride > 1, do stride conv in conv_3x3)
        self.conv2 = conv_3x3(reduced_channels, reduced_channels, stride, groups)
        self.bn2 = nn.BatchNorm2d(reduced_channels)
        
        # Use conv_1x1 to expend # channels 
        out_channels = reduced_channels * self.expansion
        self.conv3 = conv_1x1(reduced_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
        # If have stride or in_channels does not match with out_channels, identity have to do subsampling
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = downsample_1x1(in_channels, out_channels, stride)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()
        # Number of channels in each layer
        channels = [int(64 * (2 ** i)) for i in range(4)]
        self.in_channels = channels[0]
        
        # Do conv_7x7 first
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Resiudal Layers (stride=2 after second layer)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0])
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2)
        
        # Use avgpool to make the shape become (c,1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        out_channels = channels[-1] * block.expansion
        self.fc = nn.Linear(out_channels, num_classes)
        
        # Initize model weight
        self._init_weight()
        
    def _init_weight(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def _make_layer(self, block, channels, num_blocks, stride=1):
        out_channels = channels * block.expansion
        layers = []
        for i in range(num_blocks):
            # All stride are 1 except for the first one
            if i==0:
                layers.append(block(self.in_channels, channels, stride))
            else:
                layers.append(block(out_channels, channels))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # First Conv_7x7
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Residual Layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Avgpooling and Fully-Connected Layer 
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc(out)
        return out

def resnet18():
    net = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2])
    return net

def resnet50():
    net = ResNet(Bottleneck, num_blocks=[3, 4, 6, 3])
    return net


def pretrained_resnet18():
    model_ft = models.resnet18(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
    model_ft.fc = nn.Linear(512, 5)
    return model_ft

def pretrained_resnet50():
    model_ft = models.resnet50(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
    model_ft.fc = nn.Linear(2048, 5)
    return model_ft


if __name__ == '__main__':
    net = resnet50()
    print(net)