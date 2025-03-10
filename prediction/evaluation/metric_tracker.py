#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging

class MetricTracker:
    def __init__(self):
        self.train_available = False
        self.test_available = False

        # Separate running metrics for train and test
        self.running_metrics = {
            'train': self._init_metric_dict(),
            'test': self._init_metric_dict()
        }
        
        self.history = {
            'train_loss': [], 'test_loss': [],
            'train_ade': [], 'test_ade': [],
            'train_fde': [], 'test_fde': []
        }

        self.best_metrics = {'ade': float('inf'), 'epoch': 0}

    def _init_metric_dict(self):
        """Helper to initialize metrics dictionary."""
        return {key: {'value': 0, 'count': 0} for key in ['loss', 'ade', 'fde']}
    
    def update(self, metrics_dict, batch_size, phase='train'):
        """Update running metrics with batch results"""
        for key, value in metrics_dict.items():
            self.running_metrics[phase][key]['value'] += value * batch_size
            self.running_metrics[phase][key]['count'] += batch_size

    def get_averages(self, phase='train'):
        """Compute averages for specified phase."""
        if phase not in self.running_metrics:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train' or 'test'.")

        return {
            key: (metric['value'] / metric['count'] if metric['count'] > 0 else 0)
            for key, metric in self.running_metrics[phase].items()
        }

    def compute_epoch_metrics(self, phase='train'):
        """Compute and store metrics for completed epoch."""
        epoch_metrics = self.get_averages(phase)
        
        # Store epoch averages in history
        self.history[f'{phase}_loss'].append(epoch_metrics['loss'])
        self.history[f'{phase}_ade'].append(epoch_metrics['ade'])
        self.history[f'{phase}_fde'].append(epoch_metrics['fde'])

        # Reset running metrics for next epoch
        self.running_metrics[phase] = self._init_metric_dict()
        
        return epoch_metrics

    def get_current_epoch_metrics(self, phase='train'):
        """Get most recent epoch metrics."""
        if not self.history[f'{phase}_loss']:  # if history is empty
            return None
            
        return {
            'loss': self.history[f'{phase}_loss'][-1],
            'ade': self.history[f'{phase}_ade'][-1],
            'fde': self.history[f'{phase}_fde'][-1]
        }

    def get_previous_epoch_metrics(self, phase='train'):
        """Get previous epoch metrics."""
        if len(self.history[f'{phase}_loss']) < 2:  # need at least 2 epochs
            return None
            
        return {
            'loss': self.history[f'{phase}_loss'][-2],
            'ade': self.history[f'{phase}_ade'][-2],
            'fde': self.history[f'{phase}_fde'][-2]
        }
    def print_epoch_metrics(self, epoch, epochs, verbose=True):
        """Print epoch metrics including best-of-N results in a side-by-side format."""
        if not verbose:
            return

        logger = logging.getLogger('AttentionEMT')
        
        # Get current metrics from history
        train_metrics = self.get_current_epoch_metrics('train')
        test_metrics = self.get_current_epoch_metrics('test') if self.test_available else None

        # Get previous metrics for improvements
        train_prev = self.get_previous_epoch_metrics('train')
        test_prev = self.get_previous_epoch_metrics('test') if self.test_available else None

        # Header
        logger.info(f"\nEpoch [{epoch+1}/{epochs}]")
        logger.info("-" * 100)
        logger.info(f"{'Metric':12} {'Training':35} {'Validation':35}")
        logger.info("-" * 100)

        # Print metrics side by side
        for metric, name in [('loss', 'Loss'), ('ade', 'ADE'), ('fde', 'FDE')]:
            train_str = "N/A"
            val_str = "N/A"

            if train_metrics:
                train_val = train_metrics[metric]
                train_str = f"{train_val:.4f}"
                if train_prev:
                    train_imp = train_prev[metric] - train_val
                    arrow = "↓" if train_imp > 0 else "↑"
                    train_str += f" ({arrow} {abs(train_imp):.4f})"
                    # train_str += f" (↓ {train_imp:.4f})"

            if test_metrics:
                val_val = test_metrics[metric]
                val_str = f"{val_val:.4f}"
                if test_prev:
                    val_imp = test_prev[metric] - val_val
                    arrow = "↓" if val_imp > 0 else "↑"
                    val_str += f" ({arrow} {abs(val_imp):.4f})" #f" (↓ {val_imp:.4f})"

            logger.info(f"{name:12} {train_str:35} {val_str:35}")

        logger.info("-" * 100)
        
        
    def reset(self, phase='train'):
        """Reset running metrics for specified phase."""
        self.running_metrics[phase] = self._init_metric_dict()