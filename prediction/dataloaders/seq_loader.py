#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:44:12 2025

@author: nadya
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:41:48 2024

@author: nadya
"""


import torch
from torch.utils.data import Dataset
import pickle

class SeqDataset(Dataset):
    def __init__(self, data_folder, setting):
        """
        Args:
            data (list): A list of [observation, target] pairs.
                         Each element is structured as:
                         [observation: [(x, y), ...], target: [(x, y), ...]].
        """

        self.include_velocity = True
        
        file = "{}/{}.pkl".format(data_folder, setting)
        with open(file, 'rb') as f:
            self.data = pickle.load(f)
    

        # Calculate dataset statistics upon initialization
        self.mean, self.std = self._calculate_statistics()

    def _calculate_statistics(self):
        """
        Calculate mean and standard deviation of the dataset.
        Returns:
            mean (torch.Tensor): Mean of shape [4] if include_velocity else [2]
            std (torch.Tensor): Standard deviation of shape [4] if include_velocity else [2]
        """
        # Initialize list to store all sequences
        all_sequences = []
        
        # Collect all sequences
        for observation, target in self.data:
            # Convert to tensor and add to list
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            if self.include_velocity:
                # Calculate velocities for observation
                obs_vel = obs_tensor[1:] - obs_tensor[:-1]
                obs_vel = torch.cat([obs_vel[[0]], obs_vel], dim=0)
                first_target_vel = target_tensor[0:1] - obs_tensor[-1:]  # First target velocity based on last observation
                obs_tensor = torch.cat([obs_tensor, obs_vel], dim=1)
                
                # Calculate velocities for target
                rest_target_vel = target_tensor[1:] - target_tensor[:-1]  # Regular velocity computation
                target_vel = torch.cat([first_target_vel, rest_target_vel], dim=0)
                # Append velocity to target tensor
                target_tensor = torch.cat([target_tensor, target_vel], dim=1)
            
            all_sequences.append(obs_tensor)
            all_sequences.append(target_tensor)
        
        # Stack all sequences
        all_data = torch.cat(all_sequences, dim=0)  # [total_timesteps, 2 or 4]
        
        # Calculate mean and std along first dimension (timesteps)
        mean = torch.mean(all_data, dim=0)  # [2 or 4]
        std = torch.std(all_data, dim=0)    # [2 or 4]
        
        # Prevent division by zero in normalization
        std[std < 1e-6] = 1.0
        
        return mean, std


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    

    def __getitem__(self, idx):
        """
        Retrieves the observation and target for a given index.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            obs_tensor (torch.Tensor): Tensor of observation coordinates [seq_len, 2] or [seq_len, 4] if include_velocity
            target_tensor (torch.Tensor): Tensor of target coordinates [seq_len, 2] or [seq_len, 4] if include_velocity
        """
        observation, target = self.data[idx]
        
        # Convert positions to tensors
        obs_tensor = torch.tensor(observation, dtype=torch.float32)    # [seq_len, 2]
        target_tensor = torch.tensor(target, dtype=torch.float32)      # [seq_len, 2]
        
        if self.include_velocity:
            # Compute velocities for observation
            obs_vel = obs_tensor[1:] - obs_tensor[:-1]                 # [seq_len-1, 2]
            obs_vel = torch.cat([obs_vel[[0]], obs_vel], dim=0)        # [seq_len, 2]
            first_target_vel = target_tensor[0:1] - obs_tensor[-1:]    # First target velocity based on last observation
            obs_tensor = torch.cat([obs_tensor, obs_vel], dim=1)       # [seq_len, 4]

            # Calculate velocities for target
            rest_target_vel = target_tensor[1:] - target_tensor[:-1]  # Regular velocity computation
            target_vel = torch.cat([first_target_vel, rest_target_vel], dim=0)
            target_tensor = torch.cat([target_tensor, target_vel], dim=1)
            
        return obs_tensor, target_tensor