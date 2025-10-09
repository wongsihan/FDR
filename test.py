#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Few-shot multiscene fault diagnosis
"""

import torch
import numpy as np
import argparse
import os
from src.config import Config
from src.models.encoders import CNNEncoder1d, CNNEncoder2d
from src.models.relation_networks import RelationNetwork1d, RelationNetwork2d
from src.data.data_generator import pu_folders, set_seed
from src.data.dataset import get_data_loader
from src.data.data_generator import PUTask


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Few-shot multiscene fault diagnosis")
    
    # Model parameters
    parser.add_argument("-f", "--feature_dim", type=int, default=64, help="Feature dimension")
    parser.add_argument("-r", "--relation_dim", type=int, default=8, help="Relation dimension")
    parser.add_argument("-w", "--class_num", type=int, default=5, help="Number of classes")
    parser.add_argument("-s", "--sample_num_per_class", type=int, default=1, help="Samples per class")
    parser.add_argument("-u", "--hidden_unit", type=int, default=10, help="Hidden units")
    
    # Test parameters
    parser.add_argument("-t", "--test_episode", type=int, default=1000, help="Test episodes")
    
    # Hardware parameters
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device")
    
    # Data parameters
    parser.add_argument("-d", "--datatype", type=str, default='fft', choices=['fft', 'raw'], help="Data type")
    parser.add_argument("-m", "--modeltype", type=str, default='1d', choices=['1d', '2d'], help="Model type")
    parser.add_argument("-n", "--snr", type=int, default=-100, help="Signal-to-noise ratio")
    parser.add_argument("-ra", "--reduction_ratio", type=int, default=8, help="Reduction ratio")
    
    # Model paths
    parser.add_argument("--feature_encoder_path", type=str, required=True, help="Path to feature encoder model")
    parser.add_argument("--relation_network_path", type=str, required=True, help="Path to relation network model")
    
    return parser.parse_args()


def test_model(config, feature_encoder, relation_network, metatest_character_folders):
    """Test the model on test data"""
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
            sample_features = feature_encoder(torch.FloatTensor(sample_images).cuda(config.gpu))[0]
            if config.modeltype == '1d':
                sample_features = sample_features.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5 * 5)
            else:
                sample_features = sample_features.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5, 5)
            sample_features = torch.mean(sample_features, 1).squeeze(1)
            test_features = feature_encoder(torch.FloatTensor(test_images).cuda(config.gpu))[0]
            
            # Calculate relations
            if config.modeltype == '1d':
                sample_features_ext = sample_features.unsqueeze(0).repeat(config.sample_num_per_class * config.class_num, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(config.class_num, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, config.feature_dim * 2, 5 * 5)
            else:
                sample_features_ext = sample_features.unsqueeze(0).repeat(config.sample_num_per_class * config.class_num, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(config.class_num, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)
                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, config.feature_dim * 2, 5, 5)
            
            relations = relation_network(relation_pairs).view(-1, config.class_num)
            
            _, predict_labels = torch.max(relations.data, 1)
            predict_labels = predict_labels.int()
            
            rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(config.class_num * config.sample_num_per_class)]
            total_rewards += np.sum(rewards)
    
    test_accuracy = total_rewards / 1.0 / config.class_num / config.sample_num_per_class / config.test_episode
    return test_accuracy


def main():
    """Main test function"""
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = Config(
        feature_dim=args.feature_dim,
        relation_dim=args.relation_dim,
        class_num=args.class_num,
        sample_num_per_class=args.sample_num_per_class,
        hidden_unit=args.hidden_unit,
        test_episode=args.test_episode,
        gpu=args.gpu,
        datatype=args.datatype,
        modeltype=args.modeltype,
        snr=args.snr,
        reduction_ratio=args.reduction_ratio
    )
    
    # Set random seed
    set_seed(3)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    print("Loading test data...")
    
    # Load test data
    import pickle
    datapath = os.path.join(config.temp_data_path, f'{config.class_num}way.pkl')
    if os.path.exists(datapath):
        with open(datapath, 'rb') as f:
            _, metatest_character_folders = pickle.load(f)
    else:
        _, metatest_character_folders = pu_folders(config.class_num)
    
    # Initialize models
    print("Initializing models...")
    if config.modeltype == '1d':
        feature_encoder = CNNEncoder1d(config.feature_dim, config.reduction_ratio, False, config.class_num)
        relation_network = RelationNetwork1d(config.feature_dim, config.relation_dim)
    elif config.modeltype == '2d':
        feature_encoder = CNNEncoder2d(config.feature_dim)
        relation_network = RelationNetwork2d(config.feature_dim, config.relation_dim)
    
    # Load trained models
    print(f"Loading feature encoder from {args.feature_encoder_path}")
    feature_encoder.load_state_dict(torch.load(args.feature_encoder_path))
    feature_encoder.cuda(config.gpu)
    
    print(f"Loading relation network from {args.relation_network_path}")
    relation_network.load_state_dict(torch.load(args.relation_network_path))
    relation_network.cuda(config.gpu)
    
    # Test model
    print("Testing model...")
    test_accuracy = test_model(config, feature_encoder, relation_network, metatest_character_folders)
    
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    main()
