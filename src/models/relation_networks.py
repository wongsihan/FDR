#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Relation Networks for Few-shot Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationNetwork1d(nn.Module):
    """
    1D Relation Network for time series data
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(RelationNetwork1d, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size * 2, input_size, kernel_size=3, padding=1),
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
        
        self.fc1 = nn.Linear(input_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class RelationNetwork2d(nn.Module):
    """
    2D Relation Network for image data
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(RelationNetwork2d, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
