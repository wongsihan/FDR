#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data preparation script for Few-shot multiscene fault diagnosis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from src.data.data_generator import pu_folders, pu_folders_anchor, set_seed
from src.utils.logger import setlogger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare data for Few-shot multiscene fault diagnosis")
    
    parser.add_argument("-w", "--class_num", type=int, default=5, help="Number of classes")
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--output_dir", type=str, default="./tempdata", help="Output directory for processed data")
    
    return parser.parse_args()


def prepare_data(args):
    """Prepare and cache data"""
    print("=== Data Preparation for Few-shot Fault Diagnosis ===\n")
    
    # Set random seed
    set_seed(1)
    
    # Setup logging
    log_path = os.path.join(args.output_dir, 'data_preparation.log')
    os.makedirs(args.output_dir, exist_ok=True)
    setlogger(log_path)
    
    logging.info("Starting data preparation...")
    logging.info(f"Class number: {args.class_num}")
    logging.info(f"Data root: {args.data_root}")
    logging.info(f"Output directory: {args.output_dir}")
    
    try:
        # Prepare main data
        print("Preparing main dataset...")
        logging.info("Loading main dataset...")
        metatrain_character_folders, metatest_character_folders = pu_folders(args.class_num)
        
        # Save main data
        main_data_path = os.path.join(args.output_dir, f'{args.class_num}way.pkl')
        import pickle
        with open(main_data_path, 'wb') as f:
            pickle.dump((metatrain_character_folders, metatest_character_folders), f)
        
        print(f"Main dataset saved to: {main_data_path}")
        logging.info(f"Main dataset saved to: {main_data_path}")
        
        # Prepare anchor data
        print("Preparing anchor dataset...")
        logging.info("Loading anchor dataset...")
        metatrain_character_folders_anchor, metatest_character_folders_anchor = pu_folders_anchor()
        
        # Save anchor data
        anchor_data_path = os.path.join(args.output_dir, f'{args.class_num}way_anchor.pkl')
        with open(anchor_data_path, 'wb') as f:
            pickle.dump((metatrain_character_folders_anchor, metatest_character_folders_anchor), f)
        
        print(f"Anchor dataset saved to: {anchor_data_path}")
        logging.info(f"Anchor dataset saved to: {anchor_data_path}")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Training classes: {len(metatrain_character_folders)}")
        print(f"Test classes: {len(metatest_character_folders)}")
        print(f"Anchor training classes: {len(metatrain_character_folders_anchor)}")
        print(f"Anchor test classes: {len(metatest_character_folders_anchor)}")
        
        logging.info("Data preparation completed successfully!")
        print("\n=== Data Preparation Complete ===")
        
    except Exception as e:
        logging.error(f"Error during data preparation: {str(e)}")
        print(f"Error: {str(e)}")
        raise


def main():
    """Main function"""
    args = parse_args()
    prepare_data(args)


if __name__ == '__main__':
    main()
