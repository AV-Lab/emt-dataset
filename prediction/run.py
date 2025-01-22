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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


window_size = 5


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('past_trajectory', type=str, help='Past Trajectory')
    p.add_argument('future_trajectory', type=str, help='Prediction Horizon')
    p.add_argument('predictor', type=str, choices=['lstm', 'gnn', 'transformer'], help='Predictor type')
    p.add_argument('setting', type=str, choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
    p.add_argument('--annotations_path', type=str, default=None, help='If annotations are placed in a location different from recomended')
    args = p.parse_args()

    if args.setting == "evaluate" and not args.checkpoint:
        print("No checkpoint provided")
        exit
        
    # Generate setting
    ann_path = "../data/annotations/" if not args.annotations_path else args.annotations_path
    prd_ann_path = "../data/annotations/prediction_annotations"
    annotations = [prd_ann_path + '/' + f for f in os.listdir(prd_ann_path)]
    generate_prediction_annotations(raw_annotations, prd_ann_path)
    
    
    # Create DataLoader
    
    
    # Train predictor 
    
    
    # Evaluate