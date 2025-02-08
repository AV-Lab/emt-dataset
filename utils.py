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
from sklearn.model_selection import KFold

def bbox_to_xy(bbox):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center


def pad_trajectory(trajectory, target_len, prefix=True, pad_value=(0, 0)):
    if len(trajectory) < target_len:
        padding = [pad_value] * (target_len - len(trajectory))
        if prefix: return padding + trajectory
        else: return trajectory + padding
    return trajectory        
    
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


def generate_frame_setting(past_trajectory, future_trajectory, data_file, anns):
    
    records = {}
    total_length = past_trajectory+future_trajectory
    
    for ann in anns: 
        records[ann] = {}
        with open(ann, 'r') as file:
            video_records = {}
            data = json.load(file)    
            for k,v in data.items():
                frames = v['frames']
                positions = [bbox_to_xy(bbox) for bbox in v['bbox']]
                n_seq = len(positions) - total_length
                for i in range(n_seq):
                    l = i 
                    r = total_length+i
                    mid = l+past_trajectory-1
                    current_frame = frames[mid]
                    if current_frame not in video_records:
                        video_records[current_frame] = []
                    record = (k, positions[l:mid], positions[mid], positions[mid+1:r]) # object_id, past_trajectory, current_loc, future_trajectory
                    video_records[current_frame].append(record) 
            records[ann] = {k: video_records[k] for k in sorted(video_records.keys())}
    
    with open(data_file, 'wb') as f:
        pickle.dump(dict(records), f)


def generate_seq_setting(past_trajectory, future_trajectory, data_file, anns, intention=False, window_size=1):  
    trajectories = []
    
    for ann in anns: 
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
                
                    if intention:
                        intentions = v['intention'][r:r+future_trajectory]
                        trajectories.append([observation, target, intentions])
                    else:
                        trajectories.append([observation, target])
                    
    with open(data_file, 'wb') as f:
        pickle.dump(trajectories, f)

def generate_prediction_settings(past_trajectory, future_trajectory, splits, annotations, window_size, setting):
    train_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['train']]
    test_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['test']]
    
    data_folder = f"{past_trajectory}_{future_trajectory}_{window_size}"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)  
    os.makedirs(data_folder) 
    
    train_file = f"{data_folder}/train.pkl"
    test_file = f"{data_folder}/test.pkl"
    
    if setting == "seq":
        generate_seq_setting(past_trajectory, future_trajectory, train_file, train_anns, window_size=window_size)
        generate_seq_setting(past_trajectory, future_trajectory, test_file, test_anns)
    else:
        generate_frame_setting(past_trajectory, future_trajectory, train_file, train_anns)
        generate_frame_setting(past_trajectory, future_trajectory, test_file, test_anns)
    
    return data_folder
    
def generate_intention_settings(past_trajectory, future_trajectory, splits, annotations):
    train_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['train']]
    test_anns = [ann for ann in annotations if ann.split('/')[-1].split('.')[0] in splits['test']]
    
    data_folder = f"{past_trajectory}_{future_trajectory}"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)  
    os.makedirs(data_folder) 
    
    train_file = f"{data_folder}/train.pkl"
    test_file = f"{data_folder}/test.pkl"
    
    generate_seq_setting(past_trajectory, future_trajectory, train_file, train_anns, intention=True)
    generate_seq_setting(past_trajectory, future_trajectory, test_file, test_anns, intention=True)
        
    return data_folder

def generate_intention_cross_validation_settings(past_trajectory, future_trajectory, annotations, n_splits=5):
    data_folder = f"{past_trajectory}_{future_trajectory}"
    
    # If folder already exists, remove it
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)  
    os.makedirs(data_folder) 
    
    # Gather all samples
    intentions = []
    for ann in annotations: 
        with open(ann, 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                trajectory = [bbox_to_xy(bbox) for bbox in v['bbox']]
                offset = len(trajectory) - past_trajectory - future_trajectory
                num_sequences = int(offset) + 1
                for i in range(num_sequences):
                    r = i + past_trajectory
                    observation = trajectory[i:r]
                    target = trajectory[r:r+future_trajectory]
                    intention = v['intention'][r:r+future_trajectory]

                    if len(observation) < past_trajectory:
                        continue
                    if len(target) < future_trajectory:
                        continue

                    # Each item is [obs_coords, future_coords, future_intentions]
                    intentions.append([observation, target, intention])
    
    # 5-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(intentions)):
        train_data = [intentions[i] for i in train_idx]
        test_data  = [intentions[i] for i in test_idx]

        # Create a subfolder for each fold
        fold_folder = os.path.join(data_folder, f"fold_{fold_idx}")
        os.makedirs(fold_folder, exist_ok=True)

        # Save train.pkl
        with open(os.path.join(fold_folder, "train.pkl"), "wb") as f_train:
            pickle.dump(train_data, f_train)

        # Save test.pkl
        with open(os.path.join(fold_folder, "test.pkl"), "wb") as f_test:
            pickle.dump(test_data, f_test)
    
    return data_folder