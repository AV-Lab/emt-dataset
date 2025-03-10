#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List, Any
import os
import logging
import sys
from pathlib import Path

def plot_metrics(train_losses: List[float],
        test_losses: List[float],
        train_ades: List[float],
        test_ades: List[float],
        train_fdes: List[float],
        test_fdes: List[float],
        enc_seq_len: int,
        dec_seq_len: int,
        log_save_path,
        win_size,
        lr_mul
    ) -> None:
        """Plot training metrics including best-of-N predictions.
        
        Args:
            train_losses: Training loss values
            test_losses: Test loss values
            train_ades: Training ADE values
            test_ades: Test ADE values
            train_fdes: Training FDE values
            test_fdes: Test FDE values
            enc_seq_len: Encoder sequence length
            dec_seq_len: Decoder sequence length
            save_path: Path to save the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(test_losses, label='Test Loss', color='orange')
        ax1.set_title('Loss', pad=20)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ADE plot
        ax2.plot(train_ades, label='Train ADE', color='blue')
        ax2.plot(test_ades, label='Test ADE', color='orange')
        ax2.set_title('Average Displacement Error (ADE)', pad=20)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('ADE Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # FDE plot
        ax3.plot(train_fdes, label='Train FDE', color='blue')
        ax3.plot(test_fdes, label='Test FDE', color='orange')
        ax3.set_title('Final Displacement Error (FDE)', pad=20)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('FDE Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(log_save_path, exist_ok=True)
        save_file = os.path.join(log_save_path, f'training_metrics_model_{enc_seq_len}_{dec_seq_len}_W_{win_size}_lr_mul_{lr_mul}.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        
def setup_logger(log_name, name: str = 'AttentionEMT', 
                 save_path: str = None, 
                 level=logging.INFO, eval=False):
        """Set up logger configuration.
        
        Args:
            name (str): Logger name
            save_path (str): Directory to save log file
            level: Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(message)s')
        
        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(simple_formatter)
        logger.addHandler(stream_handler)
        
        # File handler if save_path is provided
        if save_path:
            if eval: 
                log_path = Path(save_path) / log_name
            else:
                 log_path = Path(save_path) / log_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path),mode='w')
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger