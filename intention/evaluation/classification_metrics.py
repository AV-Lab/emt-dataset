#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:41:48 2024

@author: nadya
"""


from sklearn.metrics import precision_score, recall_score, f1_score

def compute_precision_recall_f1(preds, targets, num_classes):
    """
    Computes both overall (weighted) and per-class precision, recall, and F1.

    Args:
        preds (Tensor): Predicted class indices of shape (batch_size, target_len).
        targets (Tensor): One-hot or raw class indices of shape (batch_size, target_len, num_classes) or (batch_size, target_len).
        num_classes (int): Total number of classes.

    Returns:
        precision (float): Overall (weighted) precision.
        recall (float): Overall (weighted) recall.
        f1 (float): Overall (weighted) F1-score.
        precision_per_class (ndarray): Precision for each class, shape (num_classes,).
        recall_per_class (ndarray): Recall for each class, shape (num_classes,).
        f1_per_class (ndarray): F1-score for each class, shape (num_classes,).
    """
    # Flatten predictions to a 1D array
    preds_flat = preds.view(-1).cpu().numpy()

    if targets.ndim == 3:
        targets_flat = targets.argmax(dim=-1).view(-1).cpu().numpy()
    else:
        targets_flat = targets.view(-1).cpu().numpy()

    # Overall (weighted) metrics
    precision = precision_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='weighted', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        targets_flat, 
        preds_flat, 
        labels=range(num_classes),
        average=None, 
        zero_division=0
    )
    recall_per_class = recall_score(
        targets_flat, 
        preds_flat, 
        labels=range(num_classes),
        average=None, 
        zero_division=0
    )
    f1_per_class = f1_score(
        targets_flat, 
        preds_flat, 
        labels=range(num_classes),
        average=None, 
        zero_division=0
    )

    return precision, recall, f1, precision_per_class, recall_per_class, f1_per_class
