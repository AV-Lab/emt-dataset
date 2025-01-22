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

def generate_setting(past_trajectory, future_trajectory, train_file, train_anns):
    data = []
        
    with open(train_file, 'wb') as f:
        pickle.dump(data, f)
    
    
    
    
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

def generate_prediction_settings(past_trajectory, future_trajectory, splits, annotations):
    train_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['train']]
    test_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['test']]
    
    data_folder = f"{past_trajectory}_{future_trajectory}"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)  
    os.makedirs(data_folder) 
    
    train_file = f"{data_folder}/train.pkl"
    generate_setting(past_trajectory, future_trajectory, train_file, train_anns)
    
    test_file = f"{data_folder}/test.pkl"
    generate_setting(past_trajectory, future_trajectory, test_file, test_anns)
    
    return data_folder
    
def generate_intention_settings():
    pass