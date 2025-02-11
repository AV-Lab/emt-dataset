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
    p.add_argument('--device', type=str, default='cuda:1', choices=['cuda', 'cpu'], help='Device to run the model on')
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
        data_folder = generate_intention_cross_validation_settings(args.past_trajectory, args.future_trajectory, annotations, n_splits=num_folds)
        
        overall_precisions, overall_recalls, overall_f1s = [], [], []
        per_class_precisions, per_class_recalls, per_class_f1s = [], [], []
        
        for fold_idx in range(num_folds):
            fold_path = os.path.join(data_folder, f"fold_{fold_idx}")
    
            train_dataset = create_dataset(fold_path, args.model_setting, setting="train")
            test_dataset = create_dataset(fold_path, args.model_setting, setting="test")
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            predictor = create_predictor(args.past_trajectory, args.future_trajectory, args.model_setting, args.device, args.normalize, args.checkpoint)
            
            predictor.train(train_loader)
    
            precision_overall, recall_overall, f1_overall, precision_per_class_fold, recall_per_class_fold, f1_per_class_fold = predictor.evaluate(test_loader)
            
            overall_precisions.append(precision_overall)
            overall_recalls.append(recall_overall)
            overall_f1s.append(f1_overall)
            
            per_class_precisions.append(precision_per_class_fold)
            per_class_recalls.append(recall_per_class_fold)
            per_class_f1s.append(f1_per_class_fold)
        
        mean_precision_overall = sum(overall_precisions) / num_folds
        mean_recall_overall = sum(overall_recalls) / num_folds
        mean_f1_overall = sum(overall_f1s) / num_folds
        
        if len(per_class_precisions) > 0:
            num_classes = len(per_class_precisions[0])
            avg_precision_per_class = [sum(fold[class_idx] for fold in per_class_precisions) / num_folds for class_idx in range(num_classes)]
            avg_recall_per_class = [sum(fold[class_idx] for fold in per_class_recalls) / num_folds for class_idx in range(num_classes)]
            avg_f1_per_class = [sum(fold[class_idx] for fold in per_class_f1s) / num_folds for class_idx in range(num_classes)]
        else:
            avg_precision_per_class = []
            avg_recall_per_class = []
            avg_f1_per_class = []
        
        print("===== 5-Fold Cross Validation Results =====")
        print(f"Mean Overall Precision: {mean_precision_overall:.4f}")
        print(f"Mean Overall Recall:    {mean_recall_overall:.4f}")
        print(f"Mean Overall F1:        {mean_f1_overall:.4f}\n")
        
        for i, (p, r, f) in enumerate(zip(avg_precision_per_class, avg_recall_per_class, avg_f1_per_class)):
            print(f"[Class {i}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
