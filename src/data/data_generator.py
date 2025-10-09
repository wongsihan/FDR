#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data generation utilities for Few-shot Learning
"""

import os
import random
import numpy as np
import torch
from scipy.io import loadmat


def set_seed(seed=1):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_class(sample):
    """Extract class name from sample path"""
    return sample.split('/')[-2]


class PUTask:
    """
    PU Task for few-shot learning
    """
    
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        
        self.train_files = []
        self.test_files = []
        self.train_labels = []
        self.test_labels = []
        
        class_folders = random.sample(self.character_folders, self.num_classes)
        index = 0
        for class_folder in class_folders:
            (file, label) = class_folder
            np.random.shuffle(file)
            self.train_files += list(file[:self.train_num, :])
            self.test_files += list(file[self.train_num:self.train_num + self.test_num, :])
            self.train_labels += [index for i in range(train_num)]
            self.test_labels += [index for i in range(test_num)]
            index += 1


def pu_folders(class_num):
    """
    Generate PU dataset folders for training and testing
    """
    # Update this path to your actual data location
    root = "./data"  # Change this to your data path
    
    # Define class labels
    labels = ['KA01', 'KA03', 'KA05', 'KA07', 'KA08', 'KI03', 'KI01', 'KI07'] + ['K001'] + ['KB23', 'KI14', 'KB27', 'KI04']
    
    random.seed(1)
    print("Loading PU dataset...")
    folds = [os.path.join(root, label, label).replace('\\', '/') for label in labels]
    
    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    
    # Load training data
    for c in folds[:-8]:
        name0 = 'N09_M07_F10_'
        name1 = c.split('/')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)).replace('\\', '/') for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]
        
        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('/')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack([file, data1])
        
        train_labels.append(get_class(temp))
        train_files.append(file)
    
    # Load testing data
    for c in folds[-5:]:
        name0 = 'N15_M01_F10_'
        name1 = c.split('/')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)).replace('\\', '/') for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]
        
        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('/')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack([file, data1])
        
        test_labels.append(get_class(temp))
        test_files.append(file)
    
    metatrain_folders = list(zip(train_files, train_labels))
    metatest_folders = list(zip(test_files, test_labels))
    return metatrain_folders, metatest_folders


def pu_folders_anchor():
    """
    Generate PU dataset folders for anchor tasks
    """
    root = "./data"  # Change this to your data path
    labels = ['KA01', 'KA03', 'KA05', 'KA07', 'KA08', 'KI03', 'KI01', 'KI07'] + ['K001'] + ['KB23', 'KI14', 'KB27', 'KI04']
    
    random.seed(1)
    print("Loading PU dataset for anchor tasks...")
    folds = [os.path.join(root, label, label).replace('\\', '/') for label in labels]
    
    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    
    # Load anchor training data
    for c in folds[0:1]:
        name0 = 'N09_M07_F10_'
        name1 = c.split('/')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)).replace('\\', '/') for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]
        
        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('/')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack([file, data1])
        
        train_labels.append(get_class(temp))
        train_files.append(file)
    
    # Load anchor testing data
    for c in folds[12:]:
        name0 = 'N15_M01_F10_'
        name1 = c.split('/')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)).replace('\\', '/') for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]
        
        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('/')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack([file, data1])
        
        test_labels.append(get_class(temp))
        test_files.append(file)
    
    metatrain_folders = list(zip(train_files, train_labels))
    metatest_folders = list(zip(test_files, test_labels))
    return metatrain_folders, metatest_folders
