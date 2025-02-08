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
from models.transformer import Attention_EMT,AttentionEMT
import torch
import torch.nn as nn

from train import train_attn,ScheduledOptim
import numpy as np
import random

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # print(f"Setting seeds: {seed}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('past_trajectory',type=int, help='Past Trajectory')
    p.add_argument('future_trajectory',type=int, help='Prediction Horizon')
    p.add_argument('epochs', type=str, default=50, help='Num of training epochs')
    p.add_argument('--window_size', default=1, type=int, help='Sliding window')
    p.add_argument('--predictor', type=str, choices=['lstm', 'gnn', 'transformer'], default='transformer',help='Predictor type')
    p.add_argument('--setting', type=str, default='train',choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
    p.add_argument('--annotations_path', type=str, default="data/annotations", help='If annotations are placed in a location different from recomended')
    p.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--device', type=str, default='cuda', help='device to run the model',choices=['cuda', 'cpu'])
    p.add_argument('--seed', type=int, default=42, help='Seed for reproducibility -> set zero for random seed generation')

    args = p.parse_args()
   

    if args.setting == "evaluate" and not args.checkpoint:
        print("No checkpoint provided")
        exit
    elif args.setting == "train" and not args.checkpoint:
        args.checkpoint = f'transformer_P_{args.past_trajectory}_F_{args.future_trajectory}_W_{args.window_size}.pth'
    if args.device == "cuda" and not cuda.is_available():
        args.device = "cpu"
        print("Could not find GPU. Using CPU instead!")

    

    # Print all arguments
    for arg in vars(args):
        if arg == "seed" and int(getattr(args, arg)) == 0:
            print(f"{arg:20s}: seed not selected (random selection)")
        else:
            print(f"{arg:20s}: {getattr(args, arg)}")


    # set seed for deterministic training -> # if seed is zero then don't set seed
    if int(args.seed)>0: 
        set_seeds(int(args.seed))

    
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
    train_dataloader = DataLoader(tain_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    test_datatset = SeqDataset(data_folder, "test",include_velocity)
    test_dataloader = DataLoader(test_datatset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)



    # get mean and standard deviation
    train_mean,train_std = tain_dataset.mean,tain_dataset.std
    max_timestep_len  =  max(args.past_trajectory, args.future_trajectory)
   


    # Train Predictor
    if args.predictor=='transformer':
        # Initialize model - > All parameters are the same as ModelConfig defaults except max_length
        transformer = AttentionEMT(
            max_length=max_timestep_len,
            device=args.device
        ).to(args.device)

        transformer.train_model(args,train_dl=train_dataloader ,test_dl=test_dataloader,epochs=int(args.epochs),mean=train_mean,std=train_std)

        
 
    
    # # Train Predictor 
    # predictor = RNNPredictor(args.past_trajectory, args.future_trajectory)
    # predictor.train(train_loader)
    # predictor.evaluate(test_loader)
    
    
    # Evaluate