#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:44:12 2025

@author: nadya
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import CLASS_LABELS

class SeqIntentionVelDataset(Dataset):
    """
    Dataset for sequential intention prediction, optionally including velocity features.
    
    Each sample consists of a past trajectory and future intentions. The past trajectory
    includes (x, y) coordinates and optionally velocity (vx, vy). Future intentions are
    provided as class indices.
    """

    def __init__(self, data_folder, setting, include_velocity=True):
        """
        Initializes the dataset by loading data from a pickle file.

        Args:
            data_folder (str): Path to the dataset directory.
            setting (str): Dataset partition ('train' or 'test').
            include_velocity (bool): If True, computes velocity features.
        """
        file_path = f"{data_folder}/{setting}.pkl"
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

        self.include_velocity = include_velocity
        self.mean, self.std = self._compute_stats_for_velocity() 

    def _compute_stats_for_velocity(self):
        """
        Computes the mean and standard deviation of velocity features across the dataset.
        
        Returns:
            mean (Tensor): Mean of the velocity features.
            std (Tensor): Standard deviation of the velocity features.
        """
        all_seqs = []
        for sample in self.data:
            observation = sample[0]
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            obs_vel = obs_tensor[1:] - obs_tensor[:-1]
            if len(obs_vel) > 0:
                obs_vel = torch.cat([obs_vel[:1], obs_vel], dim=0)
            else:
                obs_vel = torch.zeros_like(obs_tensor)
            obs_with_vel = torch.cat([obs_tensor, obs_vel], dim=1)
            all_seqs.append(obs_with_vel)

        if len(all_seqs) == 0:
            return torch.zeros(4), torch.ones(4)

        all_data = torch.cat(all_seqs, dim=0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        std[std < 1e-6] = 1.0
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset, processing the trajectory and future intentions.

        Args:
            idx (int): Index of the sample.

        Returns:
            obs_tensor (Tensor): Processed past trajectory with optional velocity.
            future_tensor (Tensor): Future intentions as class indices.
        """
        observation, _, future_intentions = self.data[idx]
        obs_tensor = torch.tensor(observation, dtype=torch.float32)

        if self.include_velocity:
            obs_vel = obs_tensor[1:] - obs_tensor[:-1]
            if len(obs_vel) > 0:
                obs_vel = torch.cat([obs_vel[:1], obs_vel], dim=0)
            else:
                obs_vel = torch.zeros_like(obs_tensor)
            obs_tensor = torch.cat([obs_tensor, obs_vel], dim=1)

            if self.mean is not None and self.std is not None:
                obs_tensor = (obs_tensor - self.mean) / self.std

        future_tensor = torch.zeros(len(future_intentions), dtype=torch.long)
        for i, intention in enumerate(future_intentions):
            cls_idx = CLASS_LABELS.get(intention, -1)
            if cls_idx < 0:
                raise ValueError(f"Unknown intention class: {intention}")
            future_tensor[i] = cls_idx

        return obs_tensor, future_tensor

####################################################################################################

class SeqOneHotVelDataset(Dataset):
    """
    Dataset for sequential intention prediction using one-hot encoded labels.
    
    Each sample consists of a past trajectory and future intentions. The past trajectory
    includes (x, y) coordinates and optionally velocity (vx, vy). Future intentions are
    represented as one-hot encoded vectors.
    """

    def __init__(self, data_folder, setting, include_velocity=True):
        """
        Initializes the dataset by loading data from a pickle file.

        Args:
            data_folder (str): Path to the dataset directory.
            setting (str): Dataset partition ('train' or 'test').
            include_velocity (bool): If True, computes velocity features.
        """
        file_path = f"{data_folder}/{setting}.pkl"
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

        self.include_velocity = include_velocity
        self.mean, self.std = self._compute_stats_for_velocity() 

    def _compute_stats_for_velocity(self):
        """
        Computes the mean and standard deviation of velocity features across the dataset.
        
        Returns:
            mean (Tensor): Mean of the velocity features.
            std (Tensor): Standard deviation of the velocity features.
        """
        all_seqs = []
        for observation, _, future_intentions in self.data:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            obs_vel = obs_tensor[1:] - obs_tensor[:-1]
            if len(obs_vel) > 0:
                obs_vel = torch.cat([obs_vel[:1], obs_vel], dim=0)
            else:
                obs_vel = torch.zeros_like(obs_tensor)
            obs_with_vel = torch.cat([obs_tensor, obs_vel], dim=1)
            all_seqs.append(obs_with_vel)

        if len(all_seqs) == 0:
            return torch.zeros(4), torch.ones(4)

        all_data = torch.cat(all_seqs, dim=0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        std[std < 1e-6] = 1.0
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset, processing the trajectory and future intentions.

        Args:
            idx (int): Index of the sample.

        Returns:
            obs_tensor (Tensor): Processed past trajectory with optional velocity.
            future_tensor (Tensor): Future intentions as one-hot encoded vectors.
        """
        observation, _, future_intentions = self.data[idx]
        obs_tensor = torch.tensor(observation, dtype=torch.float32)

        if self.include_velocity:
            obs_vel = obs_tensor[1:] - obs_tensor[:-1]
            if len(obs_vel) > 0:
                obs_vel = torch.cat([obs_vel[:1], obs_vel], dim=0)
            else:
                obs_vel = torch.zeros_like(obs_tensor)
            obs_tensor = torch.cat([obs_tensor, obs_vel], dim=1)

            if self.mean is not None and self.std is not None:
                obs_tensor = (obs_tensor - self.mean) / self.std

        future_tensor = torch.zeros(len(future_intentions), len(CLASS_LABELS), dtype=torch.float32)
        for i, intention in enumerate(future_intentions):
            cls_idx = CLASS_LABELS.get(intention, -1)
            if cls_idx < 0:
                raise ValueError(f"Unknown intention class: {intention}")
            future_tensor[i, cls_idx] = 1.0

        return obs_tensor, future_tensor
