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

class SeqToTrajectoryDataset(Dataset):
    def __init__(self, pickle_file_path, loading_parameters):
        self.pickle_file_path = pickle_file_path
        self.loading_parameters = loading_parameters
        self.load_data()  # Load metadata if needed

    def load_data(self):
        with open(self.pickle_file_path, 'rb') as file:
            trajectories = pickle.load(file)
            
        self.data = []
        for trajectory in trajectories: 
            if len(trajectory) == 2:
                t = trajectory[1]['trajectory']
            else:
                t = trajectory['trajectory']
                
            offset = len(t) - self.loading_parameters['observation_length'] - self.loading_parameters['prediction_horizon']
            num_sequences = int(offset / self.loading_parameters['sliding_window']) + 1
            for i in range(num_sequences):
                l = i * self.loading_parameters['sliding_window']
                r = l + self.loading_parameters['observation_length']
                observation = t[l:r] 
                target = t[r:r+self.loading_parameters['prediction_horizon']]
                if len(observation) < self.loading_parameters['observation_length']: continue
                if len(target) < self.loading_parameters['prediction_horizon']: continue
                self.data.append([observation, target])
                
        self.data_length = len(self.data)
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx][0], dtype=torch.float32), torch.tensor(self.data[idx][1], dtype=torch.float32))