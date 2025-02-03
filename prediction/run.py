#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:08:42 2024

@author: nadya started it but murad messed it up
"""

import os
import cv2
import sys
import argparse
import numpy as np
import json    
from torch.utils.data import DataLoader
import torch.cuda as cuda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_meta_data
from utils import generate_prediction_settings
from utils import generate_intention_settings

from evaluation.distance_metrics import calculate_ade, calculate_fde
from dataloaders.seq_loader import SeqDataset

from models.rnn import RNNPredictor
from models.transformer_model import Attention_EMT
import torch.nn as nn
import torch

from train import train_attn,ScheduledOptim

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('past_trajectory',type=int, help='Past Trajectory')
    p.add_argument('future_trajectory',type=int, help='Prediction Horizon')
    p.add_argument('--window_size', default=1, type=int, help='Sliding window')
    p.add_argument('--predictor', type=str, choices=['lstm', 'gnn', 'transformer'], default='transformer',help='Predictor type')
    p.add_argument('--setting', type=str, default='train',choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
    p.add_argument('--annotations_path', type=str, default="data/annotations", help='If annotations are placed in a location different from recomended')
    p.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--device', type=str, default='cuda', help='device to run the model',choices=['cuda', 'cpu'])
    args = p.parse_args()

    if args.setting == "evaluate" and not args.checkpoint:
        print("No checkpoint provided")
        exit
    if args.device == "cuda" and not cuda.is_available():
        args.device = "cpu"
        print("Could not find GPU. Using CPU instead!")

    

    # Print all arguments
    print("\nRunning with the following parameters:")
    for arg in vars(args):
        print(f"{arg:20s}: {getattr(args, arg)}")    

    # Generate setting
    ann_path = "../data/annotations" if not args.annotations_path else args.annotations_path
    prd_ann_path = ann_path + "/prediction_annotations"
    annotations = [prd_ann_path + '/' + f for f in os.listdir(prd_ann_path)]
    splits = load_meta_data(ann_path + "/metadata.txt")
    data_folder = generate_prediction_settings(args.past_trajectory, args.future_trajectory, splits, annotations, args.window_size)
    
    # Create DataLoaders
    # Transformer and GMM models use relative position rather than absolute position 
    if args.predictor=='transformer' or args.predictor=='gmm': 
        include_velocity = True
    else:
        include_velocity = False
        
    tain_dataset = SeqDataset(data_folder,"train",include_velocity)
    train_loader = DataLoader(tain_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    test_datatset = SeqDataset(data_folder, "test",include_velocity)
    test_loader = DataLoader(test_datatset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)


    # get mean and standard deviation
    train_mean,train_std = tain_dataset.mean,tain_dataset.std
   


    # Train Predictor
    if args.predictor=='transformer':
        # Initialize model
        transformer_model = Attention_EMT(
            in_features=2,
            out_features=2,
            num_heads=2,
            num_encoder_layers=3,
            num_decoder_layers=3,
            embedding_size=128,
            dropout=0.1,
            max_length=12,
            batch_first=True,
            actn="gelu"
        ).to(args.device)  # Move model to device
        
        # Initialize weights
        for p in transformer_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Setup optimizer parameters
        args.lr_mul = 0.1
        args.d_model = 128  # Should match embedding_size
        args.n_warmup_steps = 3500
        
        # Define the optimizer
        optimizer = ScheduledOptim(
            torch.optim.Adam(transformer_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            args.lr_mul, 
            args.d_model, 
            args.n_warmup_steps
        )

        # Train model
        trained_model, history = train_attn(
            args=args,
            train_dl=train_loader,
            test_dl=test_loader,
            model=transformer_model,  # Pass the model
            optim=optimizer,         # Pass the optimizer
            mean=train_mean,
            std=train_std,
            epochs=args.epochs if hasattr(args, 'epochs') else 10  # Add epochs parameter
        )

    
    # # Train Predictor 
    # predictor = RNNPredictor(args.past_trajectory, args.future_trajectory)
    # predictor.train(train_loader)
    # predictor.evaluate(test_loader)
    
    
    # Evaluate