#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:38:41 2025

@author: nadya
"""

import cv2
import math
import pickle
import os
import shutil 
import json
import numpy as np
import random
import torch.nn as nn
from torch import Tensor
import torch

prediction_horizon = 1

def bbox_to_xy(bbox):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center


def generate_seq_setting(past_trajectory, future_trajectory, train_file, train_anns, window_size):
    
    trajectories = []
    
    for ann in train_anns: 
        with open(ann, 'r') as file:
            data = json.load(file)    
            
            for k,v in data.items():
                trajectory = [bbox_to_xy(bbox) for bbox in v['bbox']]
                
                offset = len(trajectory) - past_trajectory - future_trajectory
                num_sequences = int(offset / window_size) + 1
                for i in range(num_sequences):
                    l = i * window_size
                    r = l + past_trajectory
                    observation = trajectory[l:r] 
                    target = trajectory[r:r+future_trajectory]
                    if len(observation) < past_trajectory: continue
                    if len(target) < future_trajectory: continue
                    trajectories.append([observation, target])
                    
    with open(train_file, 'wb') as f:
        pickle.dump(trajectories, f)
        
    
def compute_frames_idx(cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 10
    offset = 1 if input_fps % 10 == 0 else 2
    print(f"Total frames in video: {total_frames}, FPS: {input_fps}")
    
    target_frames = math.floor((total_frames * target_fps) / input_fps)
    keep_frames = set([offset + math.floor(n * input_fps / target_fps) for n in range(target_frames)])
    
    return keep_frames

def load_meta_data(file_path):
    splits = {"train": [], "test": []}
    current_section = None  # To track whether we're reading "train" or "test"

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("train:"):
                current_section = "train"
            elif line.startswith("test:"):
                current_section = "test"
            elif line:
                splits[current_section].append(line)
    
    splits['train'] = set(splits['train'])
    splits['test'] = set(splits['test'])
    return splits

def generate_prediction_settings(past_trajectory, future_trajectory, splits, annotations, window_size):
    train_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['train']]
    test_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['test']]
    
    data_folder = f"{past_trajectory}_{future_trajectory}_{window_size}"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)  
    os.makedirs(data_folder) 
    
    train_file = f"{data_folder}/train.pkl"
    test_file = f"{data_folder}/test.pkl"
    
    generate_seq_setting(past_trajectory, future_trajectory, train_file, train_anns, window_size)
    generate_seq_setting(past_trajectory, future_trajectory, test_file, test_anns, prediction_horizon)
    
    return data_folder
    
def generate_intention_settings():
    pass



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000, batch_first: bool=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if batch_first: 
            pe = torch.zeros(1,max_len, d_model)
            pe[0,:, 0::2] = torch.sin(position * div_term)
            pe[0,:, 1::2] = torch.cos(position * div_term)
        else: 
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] 
            x: Tensor, shape [batch_size, seq_len, embedding_dim]batch first
        """
        #print("pe[:,:x.size(1),:] shape: ",self.pe.shape)
        x = x + self.pe[:,:x.size(1),:] if self.batch_first else x + self.pe[:x.size(0)]

        return self.dropout(x)

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Setting seeds: {seed}")
