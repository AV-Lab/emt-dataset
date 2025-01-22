#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:41:48 2024

@author: nadya
"""


import numpy as np
import torch


def calculate_ade(predictions, targets):
    # Check if targets is a tensor
    if isinstance(targets, torch.Tensor):
        # Current implementation for tensor targets
        return torch.mean(torch.norm(predictions - targets, dim=-1)).item()
    
    # If targets is a list, convert predictions to list and compute ADE
    elif isinstance(targets, list):
        predictions_list = predictions.tolist()  # Convert predictions tensor to list
        ade = 0
        for pred, target in zip(predictions_list, targets):
            # Calculate ADE for each pair and accumulate
            ade += sum(torch.norm(torch.tensor(p) - torch.tensor(t)) for p, t in zip(pred, target))
        return ade / (len(targets) * len(targets[0]))

def calculate_fde(predictions, targets):
    # Check if targets is a tensor
    if isinstance(targets, torch.Tensor):
        # Current implementation for tensor targets
        return torch.mean(torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1)).item()
    
    # If targets is a list, convert predictions to list and compute FDE
    elif isinstance(targets, list):
        predictions_list = predictions.tolist()  # Convert predictions tensor to list
        fde = 0
        for pred, target in zip(predictions_list, targets):
            # Calculate FDE for each pair (only the last position)
            fde += torch.norm(torch.tensor(pred[-1]) - torch.tensor(target[-1]))
        return fde / len(targets)