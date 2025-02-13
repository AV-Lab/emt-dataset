#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:26:42 2025

@author: nadya
"""

import torch
from sklearn.metrics import f1_score
import Levenshtein

def compute_intention_and_distance_metrics(preds, targets, num_classes):
    """
    Computes three sets of metrics:
      1. F1-score over all timestamps (across the entire sequence) with macro average overall and per class.
      2. F1-score for the last token (intention) of each sequence, with macro average overall and per class.
      3. Average normalized Levenshtein distance between full predicted and target sequences.
      
    Args:
        preds (Tensor): Predicted class indices of shape (batch_size, seq_len).
        targets (Tensor): Ground-truth targets, either as one-hot vectors 
                          with shape (batch_size, seq_len, num_classes) or as indices with shape (batch_size, seq_len).
        num_classes (int): Total number of classes.
                          
    Returns:
        metrics (dict): A dictionary containing:
            - "f1_all_overall": Macro F1-score computed over all timestamps.
            - "f1_all_per_class": Per-class F1-score over all timestamps.
            - "f1_last_overall": Macro F1-score computed on the last token of each sequence.
            - "f1_last_per_class": Per-class F1-score on the last token.
            - "avg_norm_lev_distance": Average normalized Levenshtein distance over full sequences.
    """
    # If targets are one-hot encoded, convert them to indices.
    if targets.ndim == 3:
        targets_indices = targets.argmax(dim=-1)
    else:
        targets_indices = targets


    preds_flat = preds.cpu().numpy().flatten()
    targets_flat = targets_indices.cpu().numpy().flatten()
    f1_all_overall = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
    f1_all_per_class = f1_score(targets_flat, preds_flat, labels=range(num_classes), average=None, zero_division=0)

    preds_last = preds[:, -1].cpu().numpy()            # Shape: (batch_size,)
    targets_last = targets_indices[:, -1].cpu().numpy()  # Shape: (batch_size,)
    f1_last_overall = f1_score(targets_last, preds_last, average='macro', zero_division=0)
    f1_last_per_class = f1_score(targets_last, preds_last, labels=range(num_classes), average=None, zero_division=0)
    
    batch_size, seq_len = preds.shape
    total_norm_distance = 0.0

    for i in range(batch_size):
        # Convert each sequence of tokens to a space-separated string.
        pred_seq = " ".join(map(str, preds[i].tolist()))
        target_seq = " ".join(map(str, targets_indices[i].tolist()))
        # Compute the Levenshtein edit distance.
        edit_distance = Levenshtein.distance(pred_seq, target_seq)
        tokens_in_target = len(target_seq.split())
        norm_distance = edit_distance / tokens_in_target if tokens_in_target > 0 else 0
        total_norm_distance += norm_distance

    avg_norm_lev_distance = total_norm_distance / batch_size if batch_size > 0 else 0.0

    # Pack all metrics into a dictionary.
    metrics = {
        "f1_all_overall": f1_all_overall,
        "f1_all_per_class": f1_all_per_class,
        "f1_last_overall": f1_last_overall,
        "f1_last_per_class": f1_last_per_class,
        "avg_norm_lev_distance": avg_norm_lev_distance,
    }

    return metrics