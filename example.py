#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the Few-shot Fault Diagnosis framework
"""

import torch
import numpy as np
from src.config import Config
from src.models.encoders import CNNEncoder1d
from src.models.relation_networks import RelationNetwork1d
from src.data.data_generator import set_seed


def create_sample_data():
    """Create sample data for demonstration"""
    # Generate random time series data (simulating bearing vibration signals)
    batch_size = 10
    sequence_length = 1024
    
    # Create sample data with different patterns for different classes
    data = []
    labels = []
    
    for i in range(5):  # 5 classes
        for j in range(2):  # 2 samples per class
            # Generate synthetic vibration signal with different frequency components
            t = np.linspace(0, 1, sequence_length)
            freq1 = 50 + i * 10  # Different base frequencies for each class
            freq2 = 100 + i * 20
            
            signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
            signal += 0.1 * np.random.randn(sequence_length)  # Add noise
            
            data.append(signal)
            labels.append(i)
    
    return np.array(data), np.array(labels)


def demonstrate_model():
    """Demonstrate the model architecture"""
    print("=== Few-shot Fault Diagnosis Model Demonstration ===\n")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create configuration
    config = Config(
        feature_dim=64,
        relation_dim=8,
        class_num=5,
        sample_num_per_class=1,
        batch_num_per_class=15,
        modeltype='1d'
    )
    
    print(f"Configuration:")
    print(f"- Feature dimension: {config.feature_dim}")
    print(f"- Relation dimension: {config.relation_dim}")
    print(f"- Number of classes: {config.class_num}")
    print(f"- Model type: {config.modeltype}")
    print()
    
    # Initialize models
    print("Initializing models...")
    feature_encoder = CNNEncoder1d(
        feature_dim=config.feature_dim,
        ratio=config.reduction_ratio,
        anchor=False,
        output=config.class_num
    )
    
    relation_network = RelationNetwork1d(
        input_size=config.feature_dim,
        hidden_size=config.relation_dim
    )
    
    print(f"Feature encoder parameters: {sum(p.numel() for p in feature_encoder.parameters()):,}")
    print(f"Relation network parameters: {sum(p.numel() for p in relation_network.parameters()):,}")
    print()
    
    # Create sample data
    print("Creating sample data...")
    sample_data, sample_labels = create_sample_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample labels: {sample_labels}")
    print()
    
    # Convert to PyTorch tensors
    sample_tensor = torch.FloatTensor(sample_data).unsqueeze(1)  # Add channel dimension
    print(f"Input tensor shape: {sample_tensor.shape}")
    
    # Forward pass through feature encoder
    print("Forward pass through feature encoder...")
    with torch.no_grad():
        features, x_IN, x_useful, x_useless = feature_encoder(sample_tensor)
        print(f"Feature output shape: {features.shape}")
        print(f"Attention outputs shape: {x_IN.shape}, {x_useful.shape}, {x_useless.shape}")
    
    # Demonstrate relation network
    print("\nDemonstrating relation network...")
    # Create sample and query features
    sample_features = features[:5]  # 5 samples (one per class)
    query_features = features[5:]    # 5 queries
    
    # Reshape for relation network
    sample_features = sample_features.view(5, 1, config.feature_dim, 25)
    query_features = query_features.view(5, 1, config.feature_dim, 25)
    
    # Create relation pairs
    sample_features_ext = sample_features.unsqueeze(0).repeat(5, 1, 1, 1, 1)
    query_features_ext = query_features.unsqueeze(0).repeat(5, 1, 1, 1, 1)
    query_features_ext = torch.transpose(query_features_ext, 0, 1)
    relation_pairs = torch.cat((sample_features_ext, query_features_ext), 2).view(-1, config.feature_dim * 2, 25)
    
    print(f"Relation pairs shape: {relation_pairs.shape}")
    
    # Forward pass through relation network
    with torch.no_grad():
        relations = relation_network(relation_pairs)
        print(f"Relation scores shape: {relations.shape}")
        print(f"Relation scores: {relations.squeeze().numpy()}")
    
    print("\n=== Demonstration Complete ===")


if __name__ == '__main__':
    demonstrate_model()
