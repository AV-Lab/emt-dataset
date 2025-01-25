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
        
        file = "{}/{}.pkl".format(data_folder, setting)
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the observation and target for a given index.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            obs_tensor (torch.Tensor): Tensor of observation coordinates.
            target_tensor (torch.Tensor): Tensor of target coordinates.
        """
        observation, target = self.data[idx]
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return obs_tensor, target_tensor