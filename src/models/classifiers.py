#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Additional Classifiers for Few-shot Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


class Classifier1d(nn.Module):
    """
    1D Classifier for time series data
    """
    
    def __init__(self, input_size: int, out_size: int):
        super(Classifier1d, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.fc1 = nn.Linear(input_size * 6, out_size)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CNN1d(nn.Module):
    """
    Simple 1D CNN for time series data
    """
    
    def __init__(self, feature_dim: int):
        super(CNN1d, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU()
        )
        
        self.admp = nn.AdaptiveMaxPool1d(25)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        out = out.view(out.size(0), -1)
        return out


class DistLinear(nn.Module):
    """
    Distance-based Linear layer for few-shot learning
    """
    
    def __init__(self, indim: int, outdim: int):
        super(DistLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)
        
        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10
    
    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * cos_dist
        
        return scores
