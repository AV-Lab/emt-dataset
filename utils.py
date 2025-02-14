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

import numpy as np
import random
import torch.nn as nn
from torch import Tensor
import torch

prediction_horizon = 1


def bbox_to_xy(bbox):
    """
    Convert bounding box coordinates to the (x,y) center point.

    Args:
        bbox (list or tuple): A bounding box of the form [x_min, y_min, x_max, y_max].
    
    Returns:
        (float, float): The (x_center, y_center) of the bounding box.
    """
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility across numpy, random, and torch.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Setting seeds: {seed}")

def pad_trajectory(trajectory, target_len, prefix=True, pad_value=(0, 0)):
    """
    Pad a 2D-trajectory (list of (x,y)) to a specified length.

    Args:
        trajectory (list of tuples): The original list of (x,y) coordinates.
        target_len (int): The total length to pad/truncate to.
        prefix (bool, optional): If True, pad at the start; otherwise pad at the end.
        pad_value (tuple): The (x,y) to use for padding.

    Returns:
        list of tuples: Padded trajectory of length 'target_len'.
    """
    if len(trajectory) < target_len:
        padding = [pad_value] * (target_len - len(trajectory))
        if prefix:
            return padding + trajectory
        else:
            return trajectory + padding
    return trajectory


def compute_frames_idx(cap):
    """
    Compute the set of frame indices to keep when converting a video 
    from its original FPS to a target FPS of 10.

    Args:
        cap (cv2.VideoCapture): An open video capture object.

    Returns:
        set: A set of frame indices to keep for the target FPS conversion.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 10
    offset = 1 if input_fps % 10 == 0 else 2
    print(f"Total frames in video: {total_frames}, FPS: {input_fps}")
    
    target_frames = math.floor((total_frames * target_fps) / input_fps)
    keep_frames = set([
        offset + math.floor(n * input_fps / target_fps) for n in range(target_frames)
    ])
    
    return keep_frames


def load_meta_data(file_path):
    """
    Parse a meta-data file specifying train/test splits.

    Format example:
        train:
        <video1>
        <video2>
        ...
        test:
        <videoX>
        <videoY>
        ...

    Args:
        file_path (str): Path to the meta data text file.

    Returns:
        dict: splits = {
          'train': set(...),
          'test': set(...)
        }
        Each containing the IDs read from the file.
    """
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
    """
    Generate and save a 'frame-based' prediction setting:
      - Reads each annotation file (JSON)
      - Builds (object_id, past_trajectory, current_loc, future_trajectory)
      - Groups them by frame in a dictionary

    Args:
        past_trajectory (int): Number of past frames to consider.
        future_trajectory (int): Number of future frames to consider.
        data_file (str): Path to the output pickle file.
        anns (list of str): Paths to JSON annotation files.
    """
    records = {}
    total_length = past_trajectory + future_trajectory
    
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
                    r = total_length + i
                    mid = l + past_trajectory - 1
                    current_frame = frames[mid]
                    if current_frame not in video_records:
                        video_records[current_frame] = []
                    
                    record = (
                        k,                    # object_id
                        positions[l:mid],     # past trajectory
                        positions[mid],       # current loc
                        positions[mid+1:r]    # future trajectory
                    )
                    video_records[current_frame].append(record)
            
            records[ann] = {k: video_records[k] for k in sorted(video_records.keys())}
    
    with open(data_file, 'wb') as f:
        pickle.dump(dict(records), f)


def generate_seq_setting(past_trajectory, future_trajectory, data_file, anns, intention=False, window_size=1):
    """
    Generate and save a 'sequence-based' prediction setting:
      - Reads each annotation file
      - Extracts (past, future) segments from the full bounding box trajectory
      - Optionally includes 'intention' if provided in the JSON

    Args:
        past_trajectory (int): Number of past frames to consider in each sequence.
        future_trajectory (int): Number of future frames to consider.
        data_file (str): Output pickle file.
        anns (list of str): Paths to JSON annotation files.
        intention (bool): If True, also store intention array from the annotation.
        window_size (int): Step size to move the sliding window for sequences.

    """
    trajectories = []
    
    for ann in anns:
        with open(ann, 'r') as file:
            data = json.load(file)
            
            for k, v in data.items():
                trajectory = [bbox_to_xy(bbox) for bbox in v['bbox']]
                offset = len(trajectory) - past_trajectory - future_trajectory
                num_sequences = int(offset / window_size) + 1

                for i in range(num_sequences):
                    l = i * window_size
                    r = l + past_trajectory
                    observation = trajectory[l:r]
                    target = trajectory[r:r+future_trajectory]
                    
                    if len(observation) < past_trajectory:
                        continue
                    if len(target) < future_trajectory:
                        continue

                    if intention:
                        intentions = v['intention'][r:r+future_trajectory]
                        trajectories.append([observation, target, intentions])
                    else:
                        trajectories.append([observation, target])
                        
    print(f"Size of dataset: {len(trajectories)}")
    
    with open(data_file, 'wb') as f:
        pickle.dump(trajectories, f)


def generate_prediction_settings(past_trajectory, future_trajectory, splits, annotations, window_size, setting):
    """
    Create prediction datasets (train.pkl, test.pkl) either in 'seq' or 'frame' style.
    
    Args:
        past_trajectory (int): # past frames used
        future_trajectory (int): # future frames to predict
        splits (dict): containing 'train' and 'test' sets of annotation IDs
        annotations (list of str): paths to annotation JSONs
        window_size (int): step size if using 'seq' setting
        setting (str): 'seq' or 'frame' approach

    Returns:
        str: path to the newly created folder containing train/test pkl files
    """
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
    else:  # 'frame'
        generate_frame_setting(past_trajectory, future_trajectory, train_file, train_anns)
        generate_frame_setting(past_trajectory, future_trajectory, test_file, test_anns)
    
    return data_folder


def generate_intention_settings(past_trajectory, future_trajectory, splits, annotations):
    """
    Generate train/test data including intention labels for future frames.
    
    Args:
        past_trajectory (int): # of past frames
        future_trajectory (int): # of future frames
        splits (dict): 'train'/'test' sets of annotation IDs
        annotations (list of str): paths to JSON annotation files
    
    Returns:
        str: path to the new folder containing train.pkl and test.pkl
    """
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


def generate_intention_cross_validation_settings(past_trajectory, future_trajectory, annotations, max_observation_length, n_splits=5):
    """
    Generate cross-validation folds (train/test) including future intention labels.

    Args:
        past_trajectory (int): # of past frames
        future_trajectory (int): # of future frames
        annotations (list of str): annotation file paths
        !!!!!!max_observation_length!!!!!!!!!!!!: if generating different settings with varying past trajectory, 
            this setting makes sure that train and test samples are same but with different past trajectory  
        n_splits (int): number of cross-validation folds

    Returns:
        str: the folder name created, containing fold_0, fold_1, ...
    """
    data_folder = f"{past_trajectory}_{future_trajectory}"
    
    # If folder exists, remove it
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder)
    
    # Gather all sequences
    intentions = []
    for ann in annotations:
        with open(ann, 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                trajectory = [bbox_to_xy(bbox) for bbox in v['bbox']]
                offset = len(trajectory) - max_observation_length - future_trajectory
                num_sequences = int(offset) + 1
                for i in range(num_sequences):
                    r = i + max_observation_length
                    observation = trajectory[i:r]
                    target = trajectory[r:r+future_trajectory]
                    intention = v['intention'][r:r+future_trajectory]
                    
                    if len(observation) < max_observation_length:
                        continue
                    if len(target) < future_trajectory:
                        continue
                    
                    observation = observation[-past_trajectory:] 
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

def group_annotations_by_frame(past_trajectory, future_trajectory, annotations_file, intention=False):
    """
    Group annotation data by frame, creating records of observed past and future.

    For each frame in a video/annotation file, this function assembles:
      - The observed (x,y) positions for the past `past_trajectory` frames 
        (including the current frame).
      - The bounding boxes for those same past frames.
      - The future (x,y) positions for the next `future_trajectory` frames.
      - The future intentions (if provided) for those frames.

    Args:
        past_trajectory (int): Number of past frames to gather before the current frame.
        future_trajectory (int): Number of future frames to gather after the current frame.
        annotations_file (str): Path to the JSON file containing annotation data.
            The file is expected to have a structure where each key `k` maps to a dict
            with:
              - 'frames': list of frame indices
              - 'bbox': list of bounding boxes
              - 'intention': list of future intention values
            Each list should be the same length, describing the timeline of the object.

    Returns:
        dict:
            A dictionary keyed by frame index, where each value is a list of tuples.
            Each tuple is:
              (observed_trajectory_xy, observed_trajectory_bbox, future_trajectory_, future_intentions)
            describing the data for each object or track at that frame:
              - observed_trajectory_xy: list of (x,y) for the past frames (and current)
              - observed_trajectory_bbox: corresponding bounding boxes for those frames
              - future_trajectory_: list of (x,y) for the future frames
              - future_intentions: list of intention values for the future frames
    """
    records = {}
    with open(annotations_file, 'r') as file:
        data = json.load(file)
        for k, v in data.items():
            frames = v['frames']
            bboxs = v['bbox']
            positions = [bbox_to_xy(bbox) for bbox in bboxs]
            if intention:
                intentions = v['intention']
            
            for i, frame in enumerate(frames):
                # past frames: i - past_trajectory + 1  through i (inclusive)
                l = i - past_trajectory + 1
                l = max(l, 0)
                
                # slice for the observed trajectory
                observed_trajectory_xy = positions[l:i+1]   # include current frame
                observed_trajectory_bbox = bboxs[l:i+1]
                
                # future frames: from i+1 to i+future_trajectory
                r = i + future_trajectory + 1
                r = min(r, len(frames))
                
                future_trajectory_ = positions[i+1:r]
                
                if frame not in records:
                    records[frame] = []

                record = (observed_trajectory_xy, observed_trajectory_bbox, future_trajectory_)
                
                if intention:
                    future_intentions = intentions[i+1:r]
                    current_intention = intentions[i]
                    record = (observed_trajectory_xy, observed_trajectory_bbox, current_intention, future_trajectory_, future_intentions)
                
                records[frame].append(record)
                
                
        records = {k: records[k] for k in sorted(records.keys())}
    
    return records


