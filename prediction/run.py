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
from utils import generate_prediction_settings
from utils import generate_intention_settings

from evaluation.distance_metrics import calculate_ade, calculate_fde
from dataloaders.seq_loader import SeqDataset
from dataloaders.frame_loader import GNNDataset
from models.rnn import RNNPredictor
from models.gnn import GCNPredictor, GATPredictor

gnn_predictors = set(["gcn", "gat"])

def create_predictor(past_trajectory, future_trajectory, max_nodes, predictor):
    if predictor == "gcn":
        return GCNPredictor(past_trajectory, future_trajectory, max_nodes)
    elif predictor == "gat":
        return GATPredictor(past_trajectory, future_trajectory, max_nodes) 
    else:
        return RNNPredictor(past_trajectory, future_trajectory)
        

def create_dataset(data_folder, predictor, max_nodes, setting="train"):
    if predictor in gnn_predictors:
        return GNNDataset(data_folder, max_nodes, setting = setting)
    else:
        return SeqDataset(data_folder, setting = setting)



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('past_trajectory', type=int, help='Past Trajectory')
    p.add_argument('future_trajectory', type=int, help='Prediction Horizon')
    p.add_argument('predictor', type=str, choices=['lstm', 'gcn', 'gat', 'transformer', 'transformer_gmm'], help='Predictor type')
    p.add_argument('setting', type=str, choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
    p.add_argument('--window_size', default=1, type=int, help='Sliding window')
    p.add_argument('--max_nodes', type=int, default=40, help='Maximum number of nodes for GNN model')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
    p.add_argument('--annotations_path', type=str, default=None, help='If annotations are placed in a location different from recomended')
    args = p.parse_args()

    if args.setting == "evaluate" and not args.checkpoint:
        print("No checkpoint provided")
        exit
        
    # Generate setting
    ann_path = "../data/annotations" if not args.annotations_path else args.annotations_path
    prd_ann_path = ann_path + "/prediction_annotations"
    annotations = [prd_ann_path + '/' + f for f in os.listdir(prd_ann_path)]
    splits = load_meta_data(ann_path + "/metadata.txt")
    generating_setting = "frame" if args.predictor in gnn_predictors else "seq"
    data_folder = generate_prediction_settings(args.past_trajectory, 
                                               args.future_trajectory, 
                                               splits, annotations, 
                                               args.window_size, 
                                               generating_setting)
    
    # Create DataLoaders
    train_dataset = create_dataset(data_folder, args.predictor, args.max_nodes, setting="train")
    test_dataset = create_dataset(data_folder, args.predictor, args.max_nodes, setting="test")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train Predictor 
    predictor = create_predictor(args.past_trajectory, args.future_trajectory, args.max_nodes, args.predictor)
    predictor.train(train_loader)
    predictor.evaluate(test_loader)