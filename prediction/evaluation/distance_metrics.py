#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:41:48 2024

@author: nadya
"""


import numpy as np
import torch


def calculate_ade(predictions, targets,per_sample: bool = False):
    """
    Compute Average Displacement Error (ADE) as the mean Euclidean distance 
    between the predicted and target positions over all time steps.
    
    This function converts inputs to tensors if they are provided as lists or NumPy arrays.
    
    Args:
        predictions (torch.Tensor or list or np.ndarray): Predicted positions with shape (B, T, 2)
            or (B, T, N, 2). If a list, each element is assumed to be a tensor (or convertible)
            with shape (T, N_valid, 2).
        targets (torch.Tensor or list or np.ndarray): Ground-truth positions with the same shape as predictions.

        per_sample (bool): If True, returns a NumPy array of per-sample ADE values.
            If False (default), returns a single scalar averaged over all samples.
    
    
    Returns:
        float: The average ADE computed over all time steps and nodes.
    """
    if isinstance(predictions, list):
        total_error = 0.0
        count = 0
        per_sample_errors = []
        for pred, targ in zip(predictions, targets):
            if not isinstance(pred, torch.Tensor):
                if isinstance(pred, np.ndarray):
                    pred = torch.from_numpy(pred)
                else:
                    pred = torch.tensor(pred)
            if not isinstance(targ, torch.Tensor):
                if isinstance(targ, np.ndarray):
                    targ = torch.from_numpy(targ)
                else:
                    targ = torch.tensor(targ)

            diff = torch.norm(pred - targ, dim=-1)
            sample_error = torch.mean(diff)  # ADE for each sample
            per_sample_errors.append(sample_error)
            total_error += diff.sum().item()
            count += diff.numel()
        if per_sample:
            return per_sample_errors.cpu().numpy()
        else:
            return total_error / count if count > 0 else 0.0
    else:
        if not isinstance(predictions, torch.Tensor):
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions)
            else:
                predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            else:
                targets = torch.tensor(targets)
        if per_sample:
            # Compute per-sample errors by averaging over time (dim=1)
            per_sample_errors = torch.mean(torch.norm(predictions - targets, dim=-1), dim=1)
            return per_sample_errors.cpu().numpy()
        else:
            return torch.mean(torch.norm(predictions - targets, dim=-1)).item()


def calculate_fde(predictions, targets,per_sample: bool = False):
    """
    Compute Final Displacement Error (FDE) as the mean Euclidean distance 
    between the predicted and target positions at the final time step.
    
    This function converts inputs to tensors if they are provided as lists or NumPy arrays.
    
    Args:
        predictions (torch.Tensor or list or np.ndarray): Predicted positions with shape (B, T, 2)
            or (B, T, N, 2). If a list, each element is assumed to be a tensor (or convertible)
            with shape (T, N_valid, 2).
        targets (torch.Tensor or list or np.ndarray): Ground-truth positions with the same shape as predictions.
        per_sample (bool): If True, returns a NumPy array of per-sample FDE values.
    
    Returns:
        float: The average FDE computed over all samples (and nodes, if applicable).
    """
    if isinstance(predictions, list):
        total_error = 0.0
        count = 0
        per_sample_errors = []
        for pred, targ in zip(predictions, targets):
            if not isinstance(pred, torch.Tensor):
                if isinstance(pred, np.ndarray):
                    pred = torch.from_numpy(pred)
                else:
                    pred = torch.tensor(pred)
            if not isinstance(targ, torch.Tensor):
                if isinstance(targ, np.ndarray):
                    targ = torch.from_numpy(targ)
                else:
                    targ = torch.tensor(targ)

            diff = torch.norm(pred[-1, :] - targ[-1, :], dim=-1) 
            per_sample_errors.append(torch.mean(diff))
            total_error += diff.sum().item()
            count += diff.numel()
           
        per_sample_errors = torch.stack(per_sample_errors)
        
        if per_sample:
            return per_sample_errors.cpu().numpy()
        else:
            return total_error / count if count > 0 else 0.0
    else:
        if not isinstance(predictions, torch.Tensor):
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions)
            else:
                predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            else:
                targets = torch.tensor(targets)
        if predictions.ndim == 3:
            # Compute per-sample errors (shape: [B])
            per_sample_errors = torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1)
            if per_sample:
                return per_sample_errors.cpu().numpy()
            else:
                return torch.mean(torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1)).item()
        elif predictions.ndim == 4:
            return torch.mean(torch.norm(predictions[:, -1, :, :] - targets[:, -1, :, :], dim=-1)).item()
        else:
            raise ValueError("Unsupported tensor shape for predictions/targets.")