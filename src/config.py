#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for Few-shot multiscene fault diagnosis
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for the project"""
    
    # Model parameters
    feature_dim: int = 64
    relation_dim: int = 8
    class_num: int = 5
    sample_num_per_class: int = 1
    batch_num_per_class: int = 15
    hidden_unit: int = 10
    
    # Training parameters
    episode: int = 100
    test_episode: int = 1000
    learning_rate: float = 0.001
    test_gap: int = 5
    
    # Hardware parameters
    gpu: int = 0
    
    # Data parameters
    datatype: str = 'fft'  # 'fft' or 'raw'
    modeltype: str = '1d'  # '1d' or '2d'
    snr: int = -100  # Signal-to-noise ratio
    reduction_ratio: int = 8
    
    # Paths
    data_root: str = './data'
    result_root: str = './results'
    temp_data_path: str = './tempdata'
    temp_anchor_path: str = './tempdata_anchor'
    
    # Logging
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """Initialize paths and create directories"""
        # Create necessary directories
        os.makedirs(self.result_root, exist_ok=True)
        os.makedirs(self.temp_data_path, exist_ok=True)
        os.makedirs(self.temp_anchor_path, exist_ok=True)
        
        # Set result path
        self.result_path = os.path.join(
            self.result_root,
            f"{self.datatype}_{self.modeltype}_{self.class_num}way{self.sample_num_per_class}shot"
        )
        os.makedirs(self.result_path, exist_ok=True)
    
    def get_log_path(self) -> str:
        """Get log file path"""
        return os.path.join(self.result_path, 'train.log')
    
    def get_model_path(self, model_name: str, accuracy: Optional[float] = None) -> str:
        """Get model save path"""
        if accuracy is not None:
            filename = f"{model_name}_{accuracy:.4f}.pkl"
        else:
            filename = f"{model_name}_final.pkl"
        return os.path.join(self.result_path, filename)


# Default configuration
default_config = Config()
