#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset classes for Few-shot Learning
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import random
import os
import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft
from PIL import Image
import torchvision.transforms as transforms


class FewShotDataset(Dataset):
    """
    Base class for few-shot datasets
    """
    
    def __init__(self, task, split='train', transform=None, target_transform=None, 
                 dt='t', mt='1d', snr=None):
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.dt = dt
        self.mt = mt
        self.snr = snr
        self.image_files = self.task.train_files if self.split == 'train' else self.task.test_files
        self.labels = np.array(self.task.train_labels if self.split == 'train' else self.task.test_labels).reshape(-1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class PUDataset(FewShotDataset):
    """
    PU Bearing Dataset for few-shot learning
    """
    
    def __init__(self, *args, **kwargs):
        super(PUDataset, self).__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        image = self.image_files[idx]
        
        # Apply noise if specified
        if self.snr == -100:
            if self.dt == 'fft':
                image = abs(fft(image - np.mean(image)))[0:1024]
        else:
            image = self._noise_rw(image, self.snr)
            image = abs(fft(image - np.mean(image)))[0:1024]
        
        # Process based on model type
        if self.mt == '2d':
            image = self._noise_rw(image[0:2025], self.snr).reshape([45, 45])
            result = image
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
            im = Image.fromarray(result * 255.0)
            im = im.convert('L')
            transform = transforms.Compose([transforms.ToTensor()])
            im = transform(im)
        elif self.mt == '1d':
            image = image[0:1024].reshape([1, 1024])
        
        label = self.labels[idx]
        return image, label
    
    def _noise_rw(self, x, snr):
        """Add noise to signal"""
        if snr == -100:
            return x
        snr1 = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2, axis=0) / len(x)
        npower = xpower / snr1
        noise = np.random.normal(0, np.sqrt(npower), x.shape)
        noise_data = x + noise
        return noise_data


class ClassBalancedSampler(Sampler):
    """
    Samples 'num_inst' examples each from 'num_cl' pools
    of examples of size 'num_per_class'
    """
    
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] 
                    for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] 
                    for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]
        
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)
    
    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True, dt='t', mt='1d', snr=-100):
    """
    Get data loader for the task
    """
    dataset = PUDataset(task, split=split, transform=transforms.ToTensor(), 
                       dt=dt, mt=mt, snr=snr)
    
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
    
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    return loader
