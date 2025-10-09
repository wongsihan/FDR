#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for Few-shot multiscene fault diagnosis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import Config
from src.models.encoders import CNNEncoder1d, CNNEncoder2d
from src.models.relation_networks import RelationNetwork1d, RelationNetwork2d
from src.data.data_generator import pu_folders, set_seed, PUTask
from src.data.dataset import get_data_loader
from src.utils.logger import setlogger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Few-shot multiscene fault diagnosis model")
    
    # Model parameters
    parser.add_argument("-f", "--feature_dim", type=int, default=64, help="Feature dimension")
    parser.add_argument("-r", "--relation_dim", type=int, default=8, help="Relation dimension")
    parser.add_argument("-w", "--class_num", type=int, default=5, help="Number of classes")
    parser.add_argument("-s", "--sample_num_per_class", type=int, default=1, help="Samples per class")
    parser.add_argument("-u", "--hidden_unit", type=int, default=10, help="Hidden units")
    
    # Evaluation parameters
    parser.add_argument("-t", "--test_episode", type=int, default=1000, help="Test episodes")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of evaluation runs")
    
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
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    
    return parser.parse_args()


def load_model(config, feature_encoder_path, relation_network_path):
    """Load trained models"""
    print("Loading models...")
    
    # Initialize models
    if config.modeltype == '1d':
        feature_encoder = CNNEncoder1d(config.feature_dim, config.reduction_ratio, False, config.class_num)
        relation_network = RelationNetwork1d(config.feature_dim, config.relation_dim)
    elif config.modeltype == '2d':
        feature_encoder = CNNEncoder2d(config.feature_dim)
        relation_network = RelationNetwork2d(config.feature_dim, config.relation_dim)
    
    # Load model weights
    feature_encoder.load_state_dict(torch.load(feature_encoder_path))
    relation_network.load_state_dict(torch.load(relation_network_path))
    
    # Move to GPU
    feature_encoder.cuda(config.gpu)
    relation_network.cuda(config.gpu)
    
    # Set to evaluation mode
    feature_encoder.eval()
    relation_network.eval()
    
    print("Models loaded successfully!")
    return feature_encoder, relation_network


def evaluate_single_episode(feature_encoder, relation_network, config, metatest_character_folders):
    """Evaluate on a single episode"""
    with torch.no_grad():
        # Create test task
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
        
        # Get predictions
        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.int()
        
        # Calculate accuracy
        correct = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(config.class_num * config.sample_num_per_class)]
        accuracy = np.sum(correct) / len(correct)
        
        return accuracy, predict_labels.cpu().numpy(), test_labels


def evaluate_model(feature_encoder, relation_network, config, metatest_character_folders, num_runs=5):
    """Evaluate model with multiple runs"""
    print(f"Evaluating model with {num_runs} runs...")
    
    all_accuracies = []
    all_predictions = []
    all_labels = []
    
    for run in tqdm(range(num_runs), desc="Evaluation runs"):
        run_accuracies = []
        
        for episode in tqdm(range(config.test_episode), desc=f"Run {run+1}", leave=False):
            accuracy, predictions, labels = evaluate_single_episode(
                feature_encoder, relation_network, config, metatest_character_folders)
            run_accuracies.append(accuracy)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
        
        all_accuracies.append(run_accuracies)
    
    return all_accuracies, all_predictions, all_labels


def analyze_results(accuracies, output_dir):
    """Analyze and save evaluation results"""
    print("Analyzing results...")
    
    # Calculate statistics
    mean_accuracies = [np.mean(run) for run in accuracies]
    std_accuracies = [np.std(run) for run in accuracies]
    
    overall_mean = np.mean(mean_accuracies)
    overall_std = np.std(mean_accuracies)
    
    print(f"\nEvaluation Results:")
    print(f"Overall accuracy: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"Best run accuracy: {np.max(mean_accuracies):.4f}")
    print(f"Worst run accuracy: {np.min(mean_accuracies):.4f}")
    
    # Save results
    results = {
        'accuracies': accuracies,
        'mean_accuracies': mean_accuracies,
        'std_accuracies': std_accuracies,
        'overall_mean': overall_mean,
        'overall_std': overall_std
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for i, run_acc in enumerate(accuracies):
        plt.plot(run_acc, alpha=0.7, label=f'Run {i+1}')
    plt.title('Accuracy per Episode for Each Run')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.bar(range(1, len(mean_accuracies)+1), mean_accuracies, yerr=std_accuracies, capsize=5)
    plt.title('Mean Accuracy per Run')
    plt.xlabel('Run')
    plt.ylabel('Mean Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'))
    
    print(f"Results saved to {output_dir}")


def main():
    """Main evaluation function"""
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
    set_seed(42)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setlogger(os.path.join(args.output_dir, 'evaluation.log'))
    logging.info("Starting model evaluation...")
    
    # Load test data
    print("Loading test data...")
    datapath = os.path.join('./tempdata', f'{config.class_num}way.pkl')
    if os.path.exists(datapath):
        with open(datapath, 'rb') as f:
            _, metatest_character_folders = pickle.load(f)
    else:
        print("Test data not found. Please run data preparation first.")
        return
    
    # Load models
    feature_encoder, relation_network = load_model(config, args.feature_encoder_path, args.relation_network_path)
    
    # Evaluate model
    accuracies, predictions, labels = evaluate_model(feature_encoder, relation_network, config, 
                                                    metatest_character_folders, args.num_runs)
    
    # Analyze results
    analyze_results(accuracies, args.output_dir)
    
    logging.info("Evaluation completed!")
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
