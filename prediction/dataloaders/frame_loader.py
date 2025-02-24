#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:44:12 2025

@author: nadya
"""

import torch
import math
import pickle
from torch.utils.data import Dataset

def compute_adjacency_matrix(locations, threshold=100, normalize=True):
    """
    Compute a simple adjacency matrix based on Euclidean distance,
    optionally do symmetrical normalization:  A_hat = D^-1/2 (A+I) D^-1/2
    """
    num_nodes = len(locations)
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_ij = math.dist(locations[i], locations[j])
            if dist_ij < threshold:
                adj[i, j] = 1.0
    if normalize and num_nodes > 0:
        adj = adj + torch.eye(num_nodes)
        deg = adj.sum(dim=1) 
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt   
    return adj

class GNNDataset(Dataset):
    """
    A dataset class for a static-graph approach, optionally augmented with velocity
    for the future trajectory.
    
    Each record in 'data' is of the form (object_id, past_trajectory, current_location, future_trajectory).
    We pad each sample to a fixed maximum number of nodes (self.max_nodes), returning:
      - inputs:  (max_nodes, input_dim)
      - adj:     (max_nodes, max_nodes)
      - targets: (T_future, max_nodes, 2) or (T_future, max_nodes, 4) if include_velocity
      - mask:    (max_nodes,)
    We also compute dataset-wide mean & std for normalization, stored in self.mean, self.std.
    """
    def __init__(self, data_folder, max_nodes, setting):
        """
        Args:
            data_folder (str): folder containing <setting>.pkl
            max_nodes (int):   maximum number of nodes to pad to
            setting (str):     filename prefix, e.g. "train", "val", "test"
            include_velocity (bool): if True, future trajectory includes [x, y, vx, vy]
        """
        self.data_folder = data_folder
        self.max_nodes = max_nodes
        self.setting = setting
        self.include_velocity = True
        
        file_path = f"{data_folder}/{setting}.pkl"
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
        
        self.samples = []
        self._prepare_samples()
        
        self.mean, self.std = self._calculate_distribution_parameters()
    
    def _prepare_samples(self):
        """
        Loops over the raw data, builds (inputs, adjacency, targets, mask) for each frame,
        and appends them to self.samples.
        
        If self.include_velocity = True, we augment the future trajectory from (T_future, 2)
        to (T_future, 4): columns [0:2] = (x, y), columns [2:4] = (vx, vy).
        """
        for video_id, frames in self.data.items():
            for frame_id, records in frames.items():
                object_ids, past_trajs, curr_locs, future_trajs = map(list, zip(*records))
                
                num_nodes = len(object_ids)
                n = min(num_nodes, self.max_nodes)
                mask = torch.zeros(self.max_nodes, dtype=torch.float)
                mask[:n] = 1.0

                inputs_list = []
                for i in range(n):
                    past_tensor = torch.tensor(past_trajs[i], dtype=torch.float)  
                    curr_tensor = torch.tensor(curr_locs[i], dtype=torch.float)  
                    combined = torch.cat([past_tensor.view(-1), curr_tensor])   
                    inputs_list.append(combined)
                
                input_dim = 0
                if n > 0: input_dim = inputs_list[0].shape[0]
    
                while len(inputs_list) < self.max_nodes:
                    inputs_list.append(torch.zeros(input_dim))
                inputs_ = torch.stack(inputs_list, dim=0)  

                #Build adjency matrix
                valid_curr_locs = curr_locs[:n]  # For adjacency among valid nodes
                adj_valid = compute_adjacency_matrix(valid_curr_locs)
                adj = torch.zeros(self.max_nodes, self.max_nodes)
                adj[:n, :n] = adj_valid

                inputs_list = []
                targets_list = []
                
                for i in range(n):
                    past_tensor = torch.tensor(past_trajs[i], dtype=torch.float)  
                    future_tensor = torch.tensor(future_trajs[i], dtype=torch.float)
                    if self.include_velocity:
                        vxvy_past = past_tensor[1:] - past_tensor[:-1]
                        vxvy_past = torch.cat([vxvy_past[:1], vxvy_past], dim=0)
                        past_tensor = torch.cat([past_tensor, vxvy_past], dim=1)
                        vxvy_fut = future_tensor[1:] - future_tensor[:-1]
                        vxvy_fut = torch.cat([vxvy_fut[:1], vxvy_fut], dim=0)
                        future_tensor = torch.cat([future_tensor, vxvy_fut], dim=1)
                    inputs_list.append(past_tensor)  
                    targets_list.append(future_tensor) 
                    
                if n > 0:
                    T_past = inputs_list[0].shape[0]  
                    feat_dim_past = inputs_list[0].shape[1]  
                    T_future = targets_list[0].shape[0]
                    feat_dim_fut = targets_list[0].shape[1] 
                else:
                    T_past = 0
                    feat_dim_past = 2 if not self.include_velocity else 4
                    T_future = 0
                    feat_dim_fut = 2 if not self.include_velocity else 4
                
                if n < self.max_nodes:
                    pad_inp = torch.zeros(self.max_nodes - n, T_past, feat_dim_past)
                    inputs_ = torch.cat([torch.stack(inputs_list, dim=0), pad_inp], dim=0)
                    pad_tgt = torch.zeros(self.max_nodes - n, T_future, feat_dim_fut)
                    targets_ = torch.cat([torch.stack(targets_list, dim=0), pad_tgt], dim=0)
                else:
                    inputs_ = torch.stack(inputs_list, dim=0) 
                    targets_ = torch.stack(targets_list, dim=0) 
                
                inputs_ = inputs_.permute(1, 0, 2)  
                targets_ = targets_.permute(1, 0, 2)  
                                

                self.samples.append((inputs_, adj, targets_, mask))

    def _calculate_distribution_parameters(self):
        """
        Compute mean and std over the FUTURE data (and possibly the input data),
        depending on what you want to normalize. 
        
        Below, we show an example focusing on the future data's positions (and velocities).
        You could also gather from 'inputs' if desired.
        
        We'll gather all (T_future, max_nodes, feat_dim) from self.samples,
        reshape to a huge 2D or 3D tensor, and compute mean & std across valid nodes & timesteps.
        """
        if len(self.samples) == 0:
            # Edge case
            return torch.zeros(2), torch.ones(2)
        
        all_future = []
        for (_, _, targets, mask) in self.samples:
            feat_dim = targets.shape[-1]
            mask_3d = mask.unsqueeze(0).unsqueeze(-1).expand(targets.shape[0], targets.shape[1], feat_dim)
            valid_data = targets[mask_3d > 0].view(-1, feat_dim)  # filter out padded nodes 
            if valid_data.shape[0] > 0:
                all_future.append(valid_data)
        
        if len(all_future) == 0:
            return torch.zeros(2), torch.ones(2)
        
        all_data = torch.cat(all_future, dim=0)
        mean = all_data.mean(dim=0)
        std = all_data.std(dim=0)
        
        # Avoid tiny std
        std[std < 1e-6] = 1.0
        
        return mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
          inputs:  (max_nodes, input_dim)
          adj:     (max_nodes, max_nodes)
          targets: (T_future, max_nodes, 2 or 4)
          mask:    (max_nodes,)
        """
        return self.samples[idx]
