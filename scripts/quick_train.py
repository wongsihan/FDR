#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick training script for Few-shot multiscene fault diagnosis
This script is designed for quick testing and debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging

from src.config import Config
from src.models.encoders import CNNEncoder1d
from src.models.relation_networks import RelationNetwork1d
from src.data.data_generator import set_seed, PUTask
from src.data.dataset import get_data_loader
from src.utils.logger import setlogger


def create_synthetic_data(config):
    """Create synthetic data for quick testing"""
    print("Creating synthetic data for testing...")
    
    # Generate random time series data
    num_samples = config.class_num * 20  # 20 samples per class
    sequence_length = 1024
    
    data = []
    labels = []
    
    for class_id in range(config.class_num):
        for sample_id in range(20):
            # Generate synthetic vibration signal
            t = np.linspace(0, 1, sequence_length)
            freq1 = 50 + class_id * 10  # Different frequencies for each class
            freq2 = 100 + class_id * 20
            
            signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
            signal += 0.1 * np.random.randn(sequence_length)  # Add noise
            
            data.append(signal)
            labels.append(class_id)
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Split into train and test
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    for class_id in range(config.class_num):
        class_indices = np.where(labels == class_id)[0]
        train_indices = class_indices[:15]  # First 15 samples for training
        test_indices = class_indices[15:]   # Last 5 samples for testing
        
        train_data.extend(data[train_indices])
        train_labels.extend(labels[train_indices])
        test_data.extend(data[test_indices])
        test_labels.extend(labels[test_indices])
    
    return (np.array(train_data), np.array(train_labels), 
            np.array(test_data), np.array(test_labels))


def create_synthetic_task(train_data, train_labels, test_data, test_labels, config):
    """Create a synthetic task from the data"""
    # Group data by class
    train_files = []
    test_files = []
    
    for class_id in range(config.class_num):
        class_train_indices = np.where(train_labels == class_id)[0]
        class_test_indices = np.where(test_labels == class_id)[0]
        
        train_files.append(train_data[class_train_indices])
        test_files.append(test_data[class_test_indices])
    
    # Create task
    class_folders = list(zip(train_files, [i for i in range(config.class_num)]))
    test_folders = list(zip(test_files, [i for i in range(config.class_num)]))
    
    return class_folders, test_folders


def quick_train():
    """Quick training function for testing"""
    print("=== Quick Training for Few-shot Fault Diagnosis ===\n")
    
    # Create configuration for quick testing
    config = Config(
        feature_dim=64,
        relation_dim=8,
        class_num=5,
        sample_num_per_class=1,
        batch_num_per_class=5,  # Smaller batch for quick testing
        episode=10,            # Fewer episodes for quick testing
        test_episode=100,      # Fewer test episodes
        learning_rate=0.001,
        gpu=0,
        datatype='fft',
        modeltype='1d',
        snr=-100,
        reduction_ratio=8
    )
    
    # Set random seed
    set_seed(42)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Setup logging
    setlogger(config.get_log_path())
    logging.info("Starting quick training...")
    
    # Create synthetic data
    train_data, train_labels, test_data, test_labels = create_synthetic_data(config)
    print(f"Created synthetic data: {train_data.shape} train, {test_data.shape} test")
    
    # Create synthetic tasks
    metatrain_character_folders, metatest_character_folders = create_synthetic_task(
        train_data, train_labels, test_data, test_labels, config)
    
    # Initialize models
    print("Initializing models...")
    feature_encoder = CNNEncoder1d(config.feature_dim, config.reduction_ratio, False, config.class_num)
    feature_encoder_anchor = CNNEncoder1d(config.feature_dim, config.reduction_ratio, True, config.class_num)
    relation_network = RelationNetwork1d(config.feature_dim, config.relation_dim)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        feature_encoder.cuda(config.gpu)
        feature_encoder_anchor.cuda(config.gpu)
        relation_network.cuda(config.gpu)
        print(f"Models moved to GPU {config.gpu}")
    else:
        print("CUDA not available, using CPU")
    
    # Setup optimizers
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=config.learning_rate)
    feature_encoder_optim_anchor = torch.optim.Adam(feature_encoder_anchor.parameters(), lr=config.learning_rate)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("Starting training...")
    losses = []
    
    for episode in range(config.episode):
        # Create task
        task = PUTask(metatrain_character_folders, config.class_num, config.sample_num_per_class, config.batch_num_per_class)
        
        # Get data loaders
        sample_dataloader = get_data_loader(task, num_per_class=config.sample_num_per_class, split="train", 
                                          shuffle=False, dt=config.datatype, mt=config.modeltype)
        batch_dataloader = get_data_loader(task, num_per_class=config.batch_num_per_class, split="test", 
                                         shuffle=False, dt=config.datatype, mt=config.modeltype, snr=config.snr)
        
        # Get samples
        samples, sample_labels = sample_dataloader.__iter__().next()
        batches, batch_labels = batch_dataloader.__iter__().next()
        
        # Calculate features
        sample_features_o = feature_encoder(Variable(samples).cuda(config.gpu).float())[0]
        sample_features = sample_features_o.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5 * 5)
        sample_features = torch.mean(sample_features, 1).squeeze(1)
        batch_features = feature_encoder(Variable(batches.float()).cuda(config.gpu))[0]
        
        # Calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(config.batch_num_per_class * config.class_num, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(config.class_num, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, config.feature_dim * 2, 5 * 5)
        relations = relation_network(relation_pairs).view(-1, config.class_num)
        
        # Calculate loss
        mse = nn.MSELoss().cuda(config.gpu)
        one_hot_labels = Variable(
            torch.zeros(config.batch_num_per_class * config.class_num, config.class_num)
            .scatter_(1, batch_labels.cuda().long().view(-1, 1), 1)
        ).cuda(config.gpu)
        loss = mse(relations, one_hot_labels)
        
        # Backward pass
        feature_encoder.zero_grad()
        feature_encoder_anchor.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        
        # Update parameters
        feature_encoder_optim.step()
        feature_encoder_optim_anchor.step()
        relation_network_optim.step()
        
        losses.append(loss.item())
        print(f"Episode {episode + 1}/{config.episode}, Loss: {loss.item():.4f}")
    
    # Test the model
    print("\nTesting model...")
    feature_encoder.eval()
    relation_network.eval()
    
    total_rewards = 0
    
    with torch.no_grad():
        for i in range(config.test_episode):
            test_task = PUTask(metatest_character_folders, config.class_num, config.sample_num_per_class, config.sample_num_per_class)
            sample_dataloader = get_data_loader(test_task, num_per_class=config.sample_num_per_class, 
                                              split="train", shuffle=False, dt=config.datatype, mt=config.modeltype)
            test_dataloader = get_data_loader(test_task, num_per_class=config.sample_num_per_class, 
                                            split="test", shuffle=True, dt=config.datatype, mt=config.modeltype, snr=config.snr)
            
            sample_images, sample_labels = sample_dataloader.__iter__().next()
            test_images, test_labels = test_dataloader.__iter__().next()
            
            # Calculate features
            sample_features = feature_encoder(Variable(sample_images).cuda(config.gpu).float())[0]
            sample_features = sample_features.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5 * 5)
            sample_features = torch.mean(sample_features, 1).squeeze(1)
            test_features = feature_encoder(Variable(test_images).cuda(config.gpu).float())[0]
            
            # Calculate relations
            sample_features_ext = sample_features.unsqueeze(0).repeat(config.sample_num_per_class * config.class_num, 1, 1, 1)
            test_features_ext = test_features.unsqueeze(0).repeat(config.class_num, 1, 1, 1)
            test_features_ext = torch.transpose(test_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, config.feature_dim * 2, 5 * 5)
            relations = relation_network(relation_pairs).view(-1, config.class_num)
            
            _, predict_labels = torch.max(relations.data, 1)
            predict_labels = predict_labels.int()
            
            rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(config.class_num * config.sample_num_per_class)]
            total_rewards += np.sum(rewards)
    
    test_accuracy = total_rewards / 1.0 / config.class_num / config.sample_num_per_class / config.test_episode
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save results
    torch.save(feature_encoder.state_dict(), config.get_model_path("feature_encoder"))
    torch.save(relation_network.state_dict(), config.get_model_path("relation_network"))
    print(f"Models saved to {config.result_path}")
    
    print("\n=== Quick Training Complete ===")


if __name__ == '__main__':
    quick_train()
