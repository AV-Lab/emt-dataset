#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:08:42 2024

@author: nadya
"""

import os
import cv2
import sys
import argparse
import numpy as np
import json    
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_meta_data
from utils import generate_intention_settings
from utils import generate_intention_cross_validation_settings
from dataloaders.seq_loader import SeqIntentionVelDataset, SeqOneHotVelDataset
from models.rnn_vanilla import RNNVanillaPredictor
from models.rnn_autoregressive import RNNAutoregressivePredictor

def create_predictor(past_trajectory, future_trajectory, model_setting, device, normalize, checkpoint):
    """
    Creates an instance of the appropriate RNN-based predictor.

    Args:
        past_trajectory (int): Number of past frames to consider.
        future_trajectory (int): Number of future frames to predict.
        model_setting (str): Model type, either 'vanilla' or 'autoregressive'.
        device (str): Device to run the model on ('cuda' or 'cpu').
        normalize (bool): Whether to normalize input data.
        checkpoint (str or None): Path to a model checkpoint.

    Returns:
        Predictor instance of the specified model type.
    """
    if model_setting == "vanilla":
        return RNNVanillaPredictor(past_trajectory, future_trajectory, device, normalize, checkpoint)
    else:
        return RNNAutoregressivePredictor(past_trajectory, future_trajectory, device, normalize, checkpoint) 

def create_dataset(data_folder, model_setting, setting):
    """
    Creates a dataset based on the model setting.

    Args:
        data_folder (str): Path to the dataset folder.
        model_setting (str): Model type, either 'vanilla' or 'autoregressive'.
        setting (str): Dataset split, either 'train' or 'test'.

    Returns:
        Dataset instance corresponding to the chosen model setting.
    """
    if model_setting == "vanilla":
        return SeqOneHotVelDataset(data_folder, setting=setting)
    else:
        return SeqIntentionVelDataset(data_folder, setting=setting)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('--past_trajectory', default=10, type=int, help='Number of past frames')
    p.add_argument('--future_trajectory', default=10, type=int, help='Number of future frames to predict')
    p.add_argument('--model_setting', default="vanilla", type=str, choices=['vanilla', 'autoregressive'], help="Type of LSTM model")
    p.add_argument('--evaluation_setting', default="cross_validation", type=str, choices=['cross_validation', 'train_test'], help="Evaluation method")
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file')
    p.add_argument('--annotations_path', type=str, help='Path to annotation files if different from the default')
    p.add_argument('--num_workers', type=int, default=8, help='Number of workers for the DataLoader')
    p.add_argument('--normalize', default=False, type=bool, help='Normalize input data')
    p.add_argument('--batch_size', type=int, default=128, help='Batch size for training/testing')
    p.add_argument('--device', type=str, default='cuda:0', choices=['cuda', 'cpu'], help='Device to run the model on')
    args = p.parse_args()
        
    ann_path = "../data/annotations" if not args.annotations_path else args.annotations_path
    prd_ann_path = os.path.join(ann_path, "intention_annotations")
    annotations = [os.path.join(prd_ann_path, f) for f in os.listdir(prd_ann_path)]
    splits = load_meta_data(os.path.join(ann_path, "metadata.txt"))
    
    if args.evaluation_setting == "train_test":
        data_folder = generate_intention_settings(args.past_trajectory, args.future_trajectory, splits, annotations)
        train_dataset = create_dataset(data_folder, args.model_setting, setting="train")
        test_dataset = create_dataset(data_folder, args.model_setting, setting="test")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        predictor = create_predictor(args.past_trajectory, args.future_trajectory, args.model_setting, args.device, args.normalize, args.checkpoint)
        
        predictor.train(train_loader)
        predictor.evaluate(train_loader)
        predictor.evaluate(test_loader)
    
    else:
        num_folds = 5  
        data_folder = generate_intention_cross_validation_settings(
            args.past_trajectory, args.future_trajectory, annotations, n_splits=num_folds
        )
        
        # Lists for storing metrics for each fold.
        f1_all_overall_list   = []  # F1 over all tokens (macro overall)
        f1_last_overall_list  = []  # F1 for last token (macro overall)
        lev_distance_list     = []  # Normalized Levenshtein distance
        
        f1_all_per_class_list  = []  # Per-class F1 over all tokens
        f1_last_per_class_list = []  # Per-class F1 for last token
        
        for fold_idx in range(num_folds):
            fold_path = os.path.join(data_folder, f"fold_{fold_idx}")
        
            train_dataset = create_dataset(fold_path, args.model_setting, setting="train")
            test_dataset  = create_dataset(fold_path, args.model_setting, setting="test")
            print("reached")
            train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            predictor = create_predictor(
                args.past_trajectory, args.future_trajectory,
                args.model_setting, args.device, args.normalize, args.checkpoint
            )
            
            predictor.train(train_loader)
            metrics = predictor.evaluate(test_loader)
            
            f1_all_overall_list.append(metrics["f1_all_overall"])
            f1_last_overall_list.append(metrics["f1_last_overall"])
            lev_distance_list.append(metrics["avg_norm_lev_distance"])
            
            f1_all_per_class_list.append(metrics["f1_all_per_class"])
            f1_last_per_class_list.append(metrics["f1_last_per_class"])
        
        # Compute mean metrics over folds.
        mean_f1_all_overall  = sum(f1_all_overall_list) / num_folds
        mean_f1_last_overall = sum(f1_last_overall_list) / num_folds
        mean_lev_distance    = sum(lev_distance_list) / num_folds
        
        num_classes = len(f1_all_per_class_list[0])
        avg_f1_all_per_class  = [
            sum(fold[i] for fold in f1_all_per_class_list) / num_folds
            for i in range(num_classes)
        ]
        avg_f1_last_per_class = [
            sum(fold[i] for fold in f1_last_per_class_list) / num_folds
            for i in range(num_classes)
        ]
        
        # Print cross-validation results.
        print("===== 5-Fold Cross Validation Results =====")
        print(f"Mean F1 (All Timestamps) Overall (Macro): {mean_f1_all_overall:.4f}")
        print(f"Mean F1 (Last Token) Overall (Macro):     {mean_f1_last_overall:.4f}")
        print(f"Mean Normalized Levenshtein Distance:       {mean_lev_distance:.4f}\n")
        
        print("Per-Class F1 (All Timestamps, Macro):")
        for i, f1_val in enumerate(avg_f1_all_per_class):
            print(f"  [Class {i}]: F1: {f1_val:.4f}")
        
        print("\nPer-Class F1 (Last Token, Macro):")
        for i, f1_val in enumerate(avg_f1_last_per_class):
            print(f"  [Class {i}]: F1: {f1_val:.4f}")

