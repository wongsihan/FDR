#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main training script for Few-shot multiscene fault diagnosis
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import argparse
import logging

from src.config import Config
from src.models.encoders import CNNEncoder1d, CNNEncoder2d
from src.models.relation_networks import RelationNetwork1d, RelationNetwork2d
from src.data.data_generator import pu_folders, pu_folders_anchor, set_seed
from src.data.dataset import get_data_loader
from src.utils.logger import setlogger
from src.utils.losses import get_entropy, get_causality_loss


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Few-shot multiscene fault diagnosis")
    
    # Model parameters
    parser.add_argument("-f", "--feature_dim", type=int, default=64, help="Feature dimension")
    parser.add_argument("-r", "--relation_dim", type=int, default=8, help="Relation dimension")
    parser.add_argument("-w", "--class_num", type=int, default=5, help="Number of classes")
    parser.add_argument("-s", "--sample_num_per_class", type=int, default=1, help="Samples per class")
    parser.add_argument("-b", "--batch_num_per_class", type=int, default=15, help="Batch size per class")
    parser.add_argument("-u", "--hidden_unit", type=int, default=10, help="Hidden units")
    
    # Training parameters
    parser.add_argument("-e", "--episode", type=int, default=100, help="Number of episodes")
    parser.add_argument("-t", "--test_episode", type=int, default=1000, help="Test episodes")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
    
    # Hardware parameters
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device")
    
    # Data parameters
    parser.add_argument("-d", "--datatype", type=str, default='fft', choices=['fft', 'raw'], help="Data type")
    parser.add_argument("-m", "--modeltype", type=str, default='1d', choices=['1d', '2d'], help="Model type")
    parser.add_argument("-n", "--snr", type=int, default=-100, help="Signal-to-noise ratio")
    parser.add_argument("-ra", "--reduction_ratio", type=int, default=8, help="Reduction ratio")
    
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = Config(
        feature_dim=args.feature_dim,
        relation_dim=args.relation_dim,
        class_num=args.class_num,
        sample_num_per_class=args.sample_num_per_class,
        batch_num_per_class=args.batch_num_per_class,
        hidden_unit=args.hidden_unit,
        episode=args.episode,
        test_episode=args.test_episode,
        learning_rate=args.learning_rate,
        gpu=args.gpu,
        datatype=args.datatype,
        modeltype=args.modeltype,
        snr=args.snr,
        reduction_ratio=args.reduction_ratio
    )
    
    # Set random seed
    set_seed(3)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Setup logging
    setlogger(config.get_log_path())
    logging.info("Starting training...")
    
    # Load data
    logging.info("Loading data...")
    datapath = os.path.join(config.temp_data_path, f'{config.class_num}way.pkl')
    datapath2 = os.path.join(config.temp_anchor_path, f'{config.class_num}way.pkl')
    
    if not os.path.exists(datapath):
        metatrain_character_folders, metatest_character_folders = pu_folders(config.class_num)
        os.makedirs(config.temp_data_path, exist_ok=True)
        with open(datapath, 'wb') as f:
            pickle.dump((metatrain_character_folders, metatest_character_folders), f)
    else:
        with open(datapath, 'rb') as f:
            metatrain_character_folders, metatest_character_folders = pickle.load(f)
    
    if not os.path.exists(datapath2):
        metatrain_character_folders_anchor, metatest_character_folders_anchor = pu_folders_anchor()
        os.makedirs(config.temp_anchor_path, exist_ok=True)
        with open(datapath2, 'wb') as f:
            pickle.dump((metatrain_character_folders_anchor, metatest_character_folders_anchor), f)
    else:
        with open(datapath2, 'rb') as f:
            metatrain_character_folders_anchor, metatest_character_folders_anchor = pickle.load(f)
    
    # Initialize models
    logging.info("Initializing models...")
    if config.modeltype == '1d':
        feature_encoder = CNNEncoder1d(config.feature_dim, config.reduction_ratio, False, config.class_num)
        feature_encoder_anchor = CNNEncoder1d(config.feature_dim, config.reduction_ratio, True, config.class_num)
        relation_network = RelationNetwork1d(config.feature_dim, config.relation_dim)
    elif config.modeltype == '2d':
        feature_encoder = CNNEncoder2d(config.feature_dim)
        relation_network = RelationNetwork2d(config.feature_dim, config.relation_dim)
    
    # Move models to GPU
    feature_encoder.cuda(config.gpu)
    feature_encoder_anchor.cuda(config.gpu)
    relation_network.cuda(config.gpu)
    
    # Setup optimizers
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=config.learning_rate)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=1000, gamma=0.5)
    feature_encoder_optim_anchor = torch.optim.Adam(feature_encoder_anchor.parameters(), lr=config.learning_rate)
    feature_encoder_scheduler_anchor = StepLR(feature_encoder_optim_anchor, step_size=1000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=config.learning_rate)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=1000, gamma=0.5)
    
    # Training loop
    logging.info("Starting training...")
    last_accuracy = 0.0
    accuracys = []
    aepochs = []
    losses = []
    lepochs = []
    finalsum = 0
    
    for episode in range(config.episode):
        # Update learning rates
        feature_encoder_scheduler.step(episode)
        feature_encoder_scheduler_anchor.step(episode)
        relation_network_scheduler.step(episode)
        
        # Create tasks
        from src.data.data_generator import PUTask
        task = PUTask(metatrain_character_folders, config.class_num, config.sample_num_per_class, config.batch_num_per_class)
        task_anchor = PUTask(metatrain_character_folders_anchor, 1, config.sample_num_per_class, config.batch_num_per_class)
        task_positive = PUTask(metatrain_character_folders_anchor, 1, config.sample_num_per_class, config.batch_num_per_class)
        task_negative = PUTask(metatest_character_folders_anchor, 1, config.sample_num_per_class, config.batch_num_per_class)
        
        # Get data loaders
        sample_dataloader = get_data_loader(task, num_per_class=config.sample_num_per_class, split="train", 
                                           shuffle=False, dt=config.datatype, mt=config.modeltype)
        sample_dataloader_anchor = get_data_loader(task_anchor, num_per_class=config.sample_num_per_class, 
                                                 split="train", shuffle=False, dt=config.datatype, mt=config.modeltype)
        sample_dataloader_positive = get_data_loader(task_positive, num_per_class=config.sample_num_per_class, 
                                                   split="train", shuffle=False, dt=config.datatype, mt=config.modeltype)
        sample_dataloader_negative = get_data_loader(task_negative, num_per_class=config.sample_num_per_class, 
                                                   split="train", shuffle=False, dt=config.datatype, mt=config.modeltype)
        batch_dataloader = get_data_loader(task, num_per_class=config.batch_num_per_class, split="test", 
                                         shuffle=False, dt=config.datatype, mt=config.modeltype, snr=config.snr)
        
        # Get samples
        samples, sample_labels = sample_dataloader.__iter__().next()
        samples_anchor, sample_labels_anchor = sample_dataloader_anchor.__iter__().next()
        samples_positive, sample_labels_positive = sample_dataloader_positive.__iter__().next()
        samples_negative, sample_labels_negative = sample_dataloader_negative.__iter__().next()
        batches, batch_labels = batch_dataloader.__iter__().next()
        
        # Calculate features
        sample_features_o = feature_encoder(Variable(samples).cuda(config.gpu).float())[0]
        sample_features_o_anchor = feature_encoder_anchor(Variable(samples_anchor).cuda(config.gpu).float())[1]
        sample_features_o_positive = feature_encoder_anchor(Variable(samples_anchor).cuda(config.gpu).float())[2]
        sample_features_o_negative = feature_encoder_anchor(Variable(samples_negative).cuda(config.gpu).float())[3]
        
        if config.modeltype == '1d':
            sample_features = sample_features_o.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5 * 5)
        else:
            sample_features = sample_features_o.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5, 5)
        
        sample_features = torch.mean(sample_features, 1).squeeze(1)
        batch_features = feature_encoder(Variable(batches.float()).cuda(config.gpu))[0]
        
        # Calculate relations
        if config.modeltype == '1d':
            sample_features_ext = sample_features.unsqueeze(0).repeat(config.batch_num_per_class * config.class_num, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(config.class_num, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, config.feature_dim * 2, 5 * 5)
        else:
            sample_features_ext = sample_features.unsqueeze(0).repeat(config.batch_num_per_class * config.class_num, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(config.class_num, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, config.feature_dim * 2, 5, 5)
        
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(feature_encoder_anchor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        
        # Update parameters
        feature_encoder_optim.step()
        feature_encoder_optim_anchor.step()
        relation_network_optim.step()
        
        logging.info(f"Episode: {episode + 1}, Loss: {loss.item():.4f}")
        losses.append(loss.item())
        lepochs.append(episode)
        
        # Test every test_gap episodes
        if (episode + 1) % config.test_gap == 0:
            logging.info("Testing...")
            total_rewards = 0
            
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
                if config.modeltype == '1d':
                    sample_features = sample_features.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5 * 5)
                else:
                    sample_features = sample_features.view(config.class_num, config.sample_num_per_class, config.feature_dim, 5, 5)
                sample_features = torch.mean(sample_features, 1).squeeze(1)
                test_features = feature_encoder(Variable(test_images).cuda(config.gpu).float())[0]
                
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
            logging.info(f"Test accuracy: {test_accuracy:.4f}")
            
            if episode + 1 > 80:
                finalsum += test_accuracy
            
            accuracys.append(test_accuracy)
            aepochs.append(episode)
            
            # Save best model
            if test_accuracy > last_accuracy:
                torch.save(feature_encoder.state_dict(), config.get_model_path("feature_encoder", test_accuracy))
                torch.save(relation_network.state_dict(), config.get_model_path("relation_network", test_accuracy))
                logging.info(f"Saved best model at episode {episode + 1}")
                last_accuracy = test_accuracy
    
    # Save final models
    torch.save(feature_encoder.state_dict(), config.get_model_path("feature_encoder"))
    torch.save(relation_network.state_dict(), config.get_model_path("relation_network"))
    
    logging.info(f"Final accuracy: {test_accuracy:.4f}")
    logging.info(f"Final average accuracy: {finalsum/4:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"{config.datatype}_{config.modeltype}_{config.class_num}way{config.sample_num_per_class}shot")
    
    plt.subplot(2, 1, 1)
    plt.plot(aepochs, accuracys)
    plt.title('Test Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(lepochs, losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.result_path, 'accuracy.png'))
    
    # Save results
    with open(os.path.join(config.result_path, 'accuracy.pkl'), 'wb') as f:
        pickle.dump((aepochs, accuracys, lepochs, losses), f)
    
    logging.info("Training completed!")


if __name__ == '__main__':
    main()
