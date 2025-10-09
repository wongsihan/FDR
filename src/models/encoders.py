#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN Encoders for Few-shot Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm


class CNNEncoder1d(nn.Module):
    """
    1D CNN Encoder for time series data
    """
    
    def __init__(self, feature_dim: int, ratio: int = 8, anchor: bool = False, output: int = None):
        super(CNNEncoder1d, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3)
        self.conv1v1 = nn.Conv1d(1, 64, kernel_size=13, padding=6, stride=1)
        
        # Attention mechanism components
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(64, 64 // ratio, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(64 // ratio, 64, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Normalization layers
        self.IN1 = nn.InstanceNorm1d(64, affine=True)
        self.BN1 = nn.BatchNorm1d(64, momentum=1, affine=True)
        self.ac1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=0)
        self.IN2 = nn.InstanceNorm1d(64, affine=True)
        self.BN2 = nn.BatchNorm1d(64, momentum=1, affine=True)
        self.ac2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Third convolution layer
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm1d(64, momentum=1, affine=True)
        self.IN3 = nn.InstanceNorm1d(64, affine=True)
        self.ac3 = nn.ReLU()
        
        # Fourth convolution layer
        self.conv4 = nn.Conv1d(64, feature_dim, kernel_size=3, padding=1)
        self.BN4 = nn.BatchNorm1d(feature_dim, momentum=1, affine=True)
        self.IN4 = nn.InstanceNorm1d(feature_dim, affine=True)
        self.ac4 = nn.ReLU()
        
        # Output layer
        if anchor:
            self.fullyconnect1 = nn.Linear(64, 1)
        else:
            self.fullyconnect1 = nn.Linear(64, output)
        
        self.admp = nn.AdaptiveMaxPool1d(25)
    
    def forward(self, x):
        # First convolution with attention mechanism
        outv1 = self.conv1v1(x)
        
        # Residual connection with attention
        r1 = x - outv1
        r = self.avg_pool(r1)
        r = self.fc1(r)
        r = self.relu(r)
        r = self.fc2(r)
        a_vec = self.sigmoid(r).expand(x.size(0), 64, 1024)
        r_plus = torch.mul(a_vec, r1)
        r_sub = torch.mul((1 - a_vec), r1)
        out = outv1 + r_plus
        x_useful = out
        x_useless = outv1 + r_sub
        
        # First block
        out = self.BN1(out)
        out = self.ac1(out)
        out = self.pool1(out)
        
        # Second block
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.ac2(out)
        out = self.pool2(out)
        
        # Third block
        out = self.conv3(out)
        out = self.BN3(out)
        out = self.ac3(out)
        
        # Fourth block
        out = self.conv4(out)
        out = self.BN4(out)
        out = self.ac4(out)
        
        out = self.admp(out)
        
        # Output features for attention mechanism
        x_IN = F.softmax(self.fullyconnect1(self.avg_pool(outv1).view(outv1.size(0), -1)))
        x_useful = F.softmax(self.fullyconnect1(self.avg_pool(x_useful).view(x_useful.size(0), -1)))
        x_useless = F.softmax(self.fullyconnect1(self.avg_pool(x_useless).view(x_useless.size(0), -1)))
        
        return out, x_IN, x_useful, x_useless


class CNNEncoder2d(nn.Module):
    """
    2D CNN Encoder for image data
    """
    
    def __init__(self, feature_dim: int):
        super(CNNEncoder2d, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU()
        )
        
        self.admp = nn.AdaptiveMaxPool2d((5, 5))
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        return out


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
