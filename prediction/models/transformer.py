#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 
@author: Murdism
"""
import torch 
from torch import Tensor
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from torch.autograd import Variable # storing data while learning
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from evaluation.distance_metrics import calculate_ade,calculate_fde
from evaluation.utils import plot_metrics, setup_logger
from evaluation.metric_tracker import MetricTracker
from utils import set_seeds
from tqdm import tqdm
import os
import logging
import sys
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the AttentionEMT model."""
    # Model architecture parameters
    past_trajectory: int = 10
    future_trajectory: int = 10
    device: Optional[torch.device] = None
    normalize: bool = True
    checkpoint_file: Optional[str] = None  # Allow user-defined checkpoint
    mean: torch.tensor = torch.tensor([0.0, 0.0, 0.0, 0.0])
    std: torch.tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])
    in_features: int = 2
    out_features: int = 2
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    embedding_size: int = 128
    dropout: float = 0.2
    batch_first: bool = True
    actn: str = "gelu"
    win_size: int = 1
    

    # Optimizer parameters
    lr_mul: float = 0.2
    n_warmup_steps: int = 4000 #2000 #3000 #3500
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-9

    #loss
    criterion = nn.MSELoss()

    # Early stopping parameters
    early_stopping_patience: int = 15
    early_stopping_delta: float = 0.01

    # logging:
    log_save_path = 'results/metrics/training_metrics'

    def __post_init__(self):
        """Post-init processing."""
        # if self.checkpoint_file is None:
        #     self.checkpoint_file = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_W_x.pth'
        if self.lr_mul <= 0:
            raise ValueError("Learning rate multiplier must be positive")
        if self.n_warmup_steps < 0:
            raise ValueError("Warmup steps must be non-negative")

    def get_device(self) -> torch.device:
        """Return the device for computation."""
        return self.device if self.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def display_config(self, verbose: bool = True) -> None:
        """
        Pretty print the model configuration using logging.
        
        Args:
            verbose (bool): If True, logs additional information and formatting
        """
        logger = logging.getLogger('AttentionEMT')
        
        if verbose:
            logger.info("\n" + "="*50)
            logger.info("AttentionEMT Model Configuration")
            logger.info("="*50)
            
            logger.info("\nModel Architecture:")
            logger.info("-"*20)
            logger.info(f"Input Features:      {self.in_features}")
            logger.info(f"Output Features:     {self.out_features}")
            logger.info(f"Number of Heads:     {self.num_heads}")
            logger.info(f"Encoder Layers:      {self.num_encoder_layers}")
            logger.info(f"Decoder Layers:      {self.num_decoder_layers}")
            logger.info(f"Embedding Size:      {self.embedding_size}")
            logger.info(f"Dropout Rate:        {self.dropout}")
            logger.info(f"Batch First:         {self.batch_first}")
            logger.info(f"Activation Function: {self.actn}")
            
            logger.info("\nGMM Settings:")
            logger.info("-"*20)
            
            logger.info("\nOptimizer Settings:")
            logger.info("-"*20)
            logger.info(f"Learning Rate Multiplier: {self.lr_mul}")
            logger.info(f"Warmup Steps:            {self.n_warmup_steps}")
            logger.info(f"Optimizer Betas:         {self.optimizer_betas}")
            logger.info(f"Optimizer Epsilon:       {self.optimizer_eps}")
            
            logger.info("\nEarly Stopping Settings:")
            logger.info("-"*20)
            logger.info(f"Patience:               {self.early_stopping_patience}")
            logger.info(f"Delta:                  {self.early_stopping_delta}")
            
            logger.info("\nDevice Configuration:")
            logger.info("-"*20)
            logger.info(f"Device: {self.get_device()}")
            logger.info("\n" + "="*50)
        else:
            # Simple log of key parameters
            logger.info(
                f"AttentionEMT Config: in_features={self.in_features}, "
                f"out_features={self.out_features}, num_heads={self.num_heads}, "
                f"embedding_size={self.embedding_size}, dropout={self.dropout}, "
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)   

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        #print(self.n_warmup_steps)
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
class Linear_Embeddings(nn.Module):
    def __init__(self, input_features,d_model):
        super(Linear_Embeddings, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(input_features, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000, batch_first: bool=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if batch_first: 
            pe = torch.zeros(1,max_len, d_model)
            pe[0,:, 0::2] = torch.sin(position * div_term)
            pe[0,:, 1::2] = torch.cos(position * div_term)
        else: 
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] 
            x: Tensor, shape [batch_size, seq_len, embedding_dim]batch first
        """
        #print("pe[:,:x.size(1),:] shape: ",self.pe.shape)
        x = x + self.pe[:,:x.size(1),:] if self.batch_first else x + self.pe[:x.size(0)]

        return self.dropout(x)            
            
            
            

class AttentionEMT(nn.Module):
    """
    Attention-based Encoder-Decoder Transformer Model for time series forecasting.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_heads (int): Number of attention heads
        num_encoder_layers (int): Number of transformer encoder layers
        num_decoder_layers (int): Number of transformer decoder layers
        embedding_size (int): Size of the embedding dimension
        dropout (float): Dropout rate for encoder and decoder
        max_length (int): Maximum sequence length
        batch_first (bool): If True, batch dimension is first
        actn (str): Activation function to use
        device (torch.device): Device to use for computation
    """
    def __init__(self, config: Optional[ModelConfig] = None, checkpoint_file=None, **kwargs):
        super().__init__()

        # Create config object first
        self.config = config or ModelConfig(**kwargs)
        
        self._validate_config()
        self._init_device()
        self._init_model_params()
        self._init_layers()
        self._init_optimizer_params()
        
        if checkpoint_file is not None:
            self.load_model(checkpoint_file)
        
        self.tracker = MetricTracker()
    
    def _init_weights(self):
        """Initialize the model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.embedding_size % self.config.num_heads != 0:
            raise ValueError("Embedding size must be divisible by number of heads")
        if self.config.num_heads < 1:
            raise ValueError("Number of heads must be positive")
        if not 0 <= self.config.dropout <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.config.past_trajectory < 1 or self.config.future_trajectory < 1:
            raise ValueError("Trajectory lengths must be positive")
    def _init_device(self):
        """Initialize device configuration."""
        self.device = self.config.get_device()
        self.mean = self.config.mean.to(self.device)
        self.std = self.config.std.to(self.device)
        self.criterion = self.config.criterion
        
        
    def _init_model_params(self):
        """Initialize model parameters."""
        self.num_heads = self.config.num_heads
        self.max_len = max(self.config.past_trajectory, self.config.future_trajectory)
        self.past_trajectory = self.config.past_trajectory
        self.future_trajectory = self.config.future_trajectory
        
        self.normalized = self.config.normalize
        self.d_model = self.config.embedding_size
        self.input_features = self.config.in_features
        self.output_features = self.config.out_features
        self.dim_feedforward = 4 * self.d_model #Set feedforward dimensions (4x larger than d_model as per original paper)


        # Define dropout rates
        self.dropout_encoder = self.config.dropout
        self.dropout_decoder = self.config.dropout

        # Logging path
        self.log_save_path = self.config.log_save_path
        self.checkpoint_file = self.config.checkpoint_file
    
    def _init_optimizer_params(self):
        # Store optimizer parameters
        self.lr_mul = self.config.lr_mul 
        self.n_warmup_steps = self.config.n_warmup_steps
        self.optimizer_betas = self.config.optimizer_betas
        self.optimizer_eps = self.config.optimizer_eps

        #Initialize early stopping parameters
        self.early_stop_counter = 0
        self.early_stopping_patience = self.config.early_stopping_patience
        self.early_stopping_delta = self.config.early_stopping_delta
        self.best_metrics = {
            'ade': float('inf'),
            'fde': float('inf')
        }

    def _init_layers(self):
        """Initialize model layers."""
        # Embeddings
        self.encoder_input_layer = Linear_Embeddings(self.config.in_features, self.d_model)
        self.decoder_input_layer = Linear_Embeddings(self.config.out_features, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.max_len,
            batch_first=self.config.batch_first
        )
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.output_layer = Linear_Embeddings(self.d_model, self.output_features)
    
    def _build_encoder(self):
        """Build encoder layers."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_encoder,
            batch_first=self.config.batch_first,
            activation=self.config.actn
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_encoder_layers
        )
    
    def _build_decoder(self):
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.d_model,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout_decoder,
            batch_first = self.config.batch_first,
            activation = self.config.actn 
            )
        return nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = self.config.num_decoder_layers
        )

    
    def load_model(self, ckpt_path: str):
        """
        Load a complete model with all necessary state.
        
        Args:
            ckpt_path (str): Path to checkpoint file
        """
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(state_dict=checkpoint['model_state_dict'])
            self.to(self.device)  # Move the entire model to device
            
            # Load and move tensors to device
            self.mean = checkpoint['train_mean'].to(self.device)
            self.std = checkpoint['train_std'].to(self.device)
            
            if 'model_config' in checkpoint:
                # Update any config parameters if needed
                for key, value in checkpoint['model_config'].items():
                    setattr(self.config, key, value)
            
            return self
                
        except KeyError as e:
            raise KeyError(f"Checkpoint missing required key: {e}")
        except Exception as e:
            raise Exception(f"Error loading checkpoint: {e}")

    def configure_optimizer(
        self,
        lr_mul: Optional[float] = None,
        n_warmup_steps: Optional[int] = None,
        optimizer_betas: Optional[Tuple[float, float]] = None,
        optimizer_eps: Optional[float] = None
    ) -> ScheduledOptim:
        """
        Configure the scheduled optimizer with optional parameter overrides.
        
        Args:
            lr_mul (float, optional): Learning rate multiplier
            n_warmup_steps (int, optional): Number of warmup steps
            optimizer_betas (tuple, optional): Beta parameters for Adam
            optimizer_eps (float, optional): Epsilon parameter for Adam
            
        Returns:
            ScheduledOptim: Configured optimizer with scheduling
        """
        # Use provided values or fall back to initialization values
        lr_mul = lr_mul if lr_mul is not None else self.lr_mul
        n_warmup_steps = n_warmup_steps if n_warmup_steps is not None else self.n_warmup_steps
        optimizer_betas = optimizer_betas if optimizer_betas is not None else self.optimizer_betas
        optimizer_eps = optimizer_eps if optimizer_eps is not None else self.optimizer_eps

        # Create base optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            betas=optimizer_betas,
            eps=optimizer_eps
        )

        # Wrap with scheduler
        return ScheduledOptim(
            optimizer=optimizer,
            lr_mul=lr_mul,
            d_model=self.d_model,
            n_warmup_steps=n_warmup_steps
        )
    
    def check_early_stopping(self, current_metrics: dict, verbose: bool = True, metric='ade') -> Tuple[bool, dict]:
        """
        Check if training should stop based on the specified metric.
        
        Args:
            current_metrics (dict): Dictionary containing current metric values
            verbose (bool): Whether to print early stopping information
            metric (str): The specific metric to monitor for early stopping
            
        Returns:
            Tuple[bool, dict]: (should_stop, best_metrics)
        """
        should_stop = True
        logger = logging.getLogger('AttentionEMT')  # Changed from AttentionGMM to match your class
        
        # Only check the specified metric
        if metric in current_metrics and metric in self.best_metrics:
            current_value = current_metrics[metric]
            
            # Check if the current value is better than the best value
            if current_value < (self.best_metrics[metric] + self.config.early_stopping_delta):
                self.best_metrics[metric] = current_value
                should_stop = False
        
        # Update counter based on improvement
        if should_stop:
            self._early_stop_counter += 1
            if verbose and self._early_stop_counter > 0:
                logger.info(f"\nNo improvement in {metric} for {self._early_stop_counter} epochs.")
        else:
            self._early_stop_counter = 0
        
        # Check if we should stop training
        should_stop = self._early_stop_counter >= self.config.early_stopping_patience
        
        # Log early stopping information if triggered
        if should_stop and verbose:
            logger.info(f"\nEarly stopping triggered after {self._early_stop_counter} epochs without improvement in {metric}")
            logger.info(f"Best {metric.upper()}: {self.best_metrics[metric]:.4f}")
        
        return should_stop, self.best_metrics.copy()
    
    def _save_checkpoint(self,optimizer, epoch=10,save_model=True,save_frequency=10,save_path="/results"):
        # Set up directory structure
        models_dir = os.path.join(save_path, 'pretrained_models')
        os.makedirs(models_dir, exist_ok=True)
        logger = logging.getLogger('AttentionEMT')
        
        if save_model and (epoch + 1) % save_frequency == 0:
            model_state = {
                'model_state_dict': self.state_dict(),  # Save directly
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'training_history': self.tracker.history,
                'best_metrics': self.tracker.best_metrics,
                'train_mean': self.mean,
                'train_std': self.std,
                'model_config': {
                    # Only save what you actually use for loading
                    'in_features': self.input_features,
                    'out_features': self.output_features,
                    'num_heads': self.num_heads,
                    'num_encoder_layers': self.config.num_encoder_layers,
                    'num_decoder_layers': self.config.num_decoder_layers,
                    'embedding_size': self.d_model,
                    'dropout': self.dropout_encoder
                }
            }
            # Save the model
            checkpoint_name = f'Transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_Warm_{self.n_warmup_steps}_W_{self.config.win_size}_lr_mul_{self.lr_mul}_epoch_{epoch}.pth'
            os.makedirs(save_path, exist_ok=True)
            torch.save(model_state, os.path.join(models_dir, f"{checkpoint_name}"))
            logger.info(f"Saved checkpoint to: {save_path}")
            
    def _generate_square_mask(
        self,
        dim_trg: int,
        dim_src: int,
        mask_type: str = "tgt"
    ) -> torch.Tensor:
        """
        Generate a square mask for transformer attention mechanisms.
        
        Args:
            dim_trg (int): Target sequence length.
            dim_src (int): Source sequence length.
            mask_type (str): Type of mask to generate. Can be "src", "tgt", or "memory".
        
        Returns:
            torch.Tensor: A mask tensor with `-inf` values to block specific positions.
        """
        # Initialize a square matrix filled with -inf (default to a fully masked state)
        mask = torch.ones(dim_trg, dim_trg) * float('-inf')

        if mask_type == "src":
            # Source mask (self-attention in the encoder)
            # Creates an upper triangular matrix with -inf above the diagonal
            mask = torch.triu(mask, diagonal=1)

        elif mask_type == "tgt":
            # Target mask (self-attention in the decoder)
            # Prevents the decoder from attending to future tokens
            mask = torch.triu(mask, diagonal=1)

        elif mask_type == "memory":
            # Memory mask (cross-attention between encoder and decoder)
            # Controls which encoder outputs the decoder can attend to
            mask = torch.ones(dim_trg, dim_src) * float('-inf')
            mask = torch.triu(mask, diagonal=1)  # Prevents attending to future positions

        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            src (torch.Tensor): Source sequence
            tgt (torch.Tensor): Target sequence
            src_mask (torch.Tensor, optional): Mask for source sequence
            tgt_mask (torch.Tensor, optional): Mask for target sequence
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Add input validation
        if src.dim() != 3 or tgt.dim() != 3:
            raise ValueError("Expected 3D tensors for src and tgt")
        
        # Move inputs to device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self.device)
        
        # Encoder forward pass
        encoder_embed = self.encoder_input_layer(src)
        encoder_embed = self.positional_encoding(encoder_embed)
        encoder_output = self.encoder(src=encoder_embed)
        
        # Decoder forward pass
        decoder_embed = self.decoder_input_layer(tgt)
        decoder_embed = self.positional_encoding(decoder_embed)
        decoder_output = self.decoder(
            tgt=decoder_embed,
            memory=encoder_output,
            tgt_mask=tgt_mask
        )
        
        # Output projection
        output = self.output_layer(decoder_output) 
        return output
    
    def train(
        self,
        train_dl: DataLoader,
        test_dl: DataLoader = None,
        epochs: int = 85,
        verbose: bool = True,
        save_path: str = 'results',
        save_model: bool = True,
        save_frequency: int = 80,
    ) -> Tuple[nn.Module, Dict]:
        """
        Train the model with metrics tracking and visualization.
        """
        # Setup logger
        logger = logging.getLogger('AttentionEMT')
        if not logger.handlers:
            log_name = f'transformer_training_metrics_model_{self.past_trajectory}_{self.future_trajectory}_training_{self.n_warmup_steps}_W_{self.config.win_size}_lr_mul_{self.lr_mul}.log'
            logger = setup_logger(log_name, save_path=self.log_save_path)
        
        self.to(self.device)
        self._init_weights()
        
        #  if verbose print config:
        self.config.display_config(verbose)

        # Setup optimizer with model's configuration
        optimizer = self.configure_optimizer(
            lr_mul=self.lr_mul,
            n_warmup_steps=self.n_warmup_steps,
            optimizer_betas=self.optimizer_betas,
            optimizer_eps=self.optimizer_eps
        )

        # set metrics tracker:
        self.tracker.train_available = True
        
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        # get mean and standard deviation from training dataset
        self.mean= train_dl.dataset.mean.to(self.device)
        self.std = train_dl.dataset.std.to(self.device)
        for epoch in range(epochs):
            super().train()  # Set train mode again for safety

            # Training loop with progress bar
            load_train = tqdm(train_dl, desc=f"Epoch: {epoch+1}/{epochs}") if verbose else train_dl

            for id_b, batch in enumerate(load_train):
                # Prepare input data
                obs_tensor, target_tensor = batch
                batch_size, enc_seq_len, feat_dim = obs_tensor.shape
                dec_seq_len = target_tensor.shape[1]
                
                # Move to device and normalize
                obs_tensor = obs_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                input_train = (obs_tensor[:,1:,2:4] - self.mean[2:])/self.std[2:]
                updated_enq_length = input_train.shape[1]
                target = ((target_tensor[:, :, 2:4] - self.mean[2:]) / self.std[2:]).clone()

                # Prepare target input (teacher forcing)
                tgt = torch.zeros_like(target).to(self.device)
                tgt[:, 1:, :] = target[:, :-1, :].detach() # Detach for memory efficiency

                # Generate masks
                tgt_mask = self._generate_square_mask(
                    dim_trg=dec_seq_len,
                    dim_src=updated_enq_length,
                    mask_type="tgt"
                ).to(self.device)
                

                # Forward pass
                optimizer.zero_grad()
                pred = self(input_train, tgt, tgt_mask=tgt_mask)
                
                # Calculate loss
                train_loss = self.criterion(pred, target)
                
                # Backward pass
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                optimizer.step_and_update_lr()

                # Calculate metrics
                with torch.no_grad():
                    obs_last_pos = obs_tensor[:, -1:, 0:2]
                    mad, fad = self.calculate_metrics(
                        pred.detach(), target.detach(), obs_last_pos)
                    # Update metrics using tracker
                    batch_metrics = {
                        'loss': train_loss.item(),
                        'ade': mad,
                        'fde': fad
                    }
                    
                    self.tracker.update(batch_metrics, obs_tensor.shape[0], phase='train')
                    #Update progress bar
                    if verbose:
                        train_avgs = self.tracker.get_averages('train')
                        load_train.set_postfix({
                            'Loss': f"{train_avgs['loss']:.4f}",
                            'lr': f"{optimizer._optimizer.param_groups[0]['lr']:.6f}",
                            'ADE': f"{train_avgs['ade']:.4f}",
                            'FDE': f"{train_avgs['fde']:.4f}"
                        })
            
            # At end of epoch
            self.tracker.compute_epoch_metrics(phase='train')
            
            # Test evaluation
            if test_dl is not None and (epoch+1)%5 ==0:
                self.evaluate(test_dl,from_train=True)

            # Print epoch metrics
            self.tracker.print_epoch_metrics(epoch, epochs, verbose)

            # Check early stopping conditions
            phase = 'test' if test_dl else 'train'
            current_metrics = {
                'loss': self.tracker.get_averages(phase)['loss'],
                'ade': self.tracker.get_averages(phase)['ade'],
                'fde': self.tracker.get_averages(phase)['fde']
            }
            
            should_stop, best_metrics = self.check_early_stopping(current_metrics, verbose,metric='ade')

            # Save model if save_frequency reached 
            self._save_checkpoint(optimizer, epoch,save_model,save_frequency,save_path)
            
            # Break if early stopping triggered
            if should_stop:
                logger.info("Early stopping triggered. Ending training.")
                break

            # Reset metrics for next epoch
            self.tracker.reset('train')
            self.tracker.reset('test')

        # Plot training history if verbose
        if verbose:
            plot_metrics(
                self.tracker.history['train_loss'],
                self.tracker.history['test_loss'],
                self.tracker.history['train_ade'],
                self.tracker.history['test_ade'],
                self.tracker.history['train_fde'],
                self.tracker.history['test_fde'],
                enc_seq_len,
                dec_seq_len,
                self.log_save_path,
                self.config.win_size,
                self.lr_mul
            )
            logger.info(f"Training plots saved to {metrics_dir}")
        
        return self, self.tracker.history
    

            
    def evaluate(self, test_loader=None, ckpt_path=None, from_train=False):
        """
        Evaluate the model on test data
        Args:
            test_loader: DataLoader for test data
            ckpt_path: Path to checkpoint file
            from_train: Boolean indicating if called during training
        """
        logger = logging.getLogger('AttentionEMT')
    
        if test_loader is None:
            raise ValueError("test_loader cannot be None")
            
        # Store initial training mode
        training = self.training
        
        try:
            if not from_train:
                self.load_model(self.checkpoint_file)  
                # Setup logger
                logger = logging.getLogger('AttentionEMT')
                if not logger.handlers:
                    log_name = f'eval_{self.past_trajectory}_{self.future_trajectory}_W_{self.config.win_size}.log' 
                    logger = setup_logger(log_name, save_path=self.log_save_path,eval=True)                                                   
           
            # Set evaluation mode :  Need to use nn.Module's train method for mode setting
            super().train(False)  
            self.tracker.test_available = True
            
            with torch.no_grad():
                for batch in test_loader:
                    obs_tensor_eval, target_tensor_eval = batch
                    
                    # dimension check
                    assert obs_tensor_eval.shape[-1] == 4, "Expected input with 4 features (pos_x, pos_y, vel_x, vel_y)"
                    
                    obs_tensor_eval = obs_tensor_eval.to(self.device)
                    target_tensor_eval = target_tensor_eval.to(self.device)
                    dec_seq_len = target_tensor_eval.shape[1]

                    input_eval = (obs_tensor_eval[:,1:,2:4] - self.mean[2:])/self.std[2:]
                    updated_enq_length = input_eval.shape[1]
                    target_eval = ((target_tensor_eval[:, :, 2:4] - self.mean[2:]) / self.std[2:]).clone()


                    # Initialize first decoder input as zeros
                    tgt_eval = torch.zeros_like(target_eval).to(self.device)
            
                    # Autoregressive generation
                    for t in range(target_eval.shape[1]-1):
                        tgt_mask = self._generate_square_mask(dim_trg=dec_seq_len, 
                                                    dim_src=updated_enq_length, 
                                                    mask_type="tgt").to(self.device)
                        
                        eval_pred = self(input_eval, tgt_eval, tgt_mask=tgt_mask)
                        # Use the current prediction as input for next timestep
                        tgt_eval[:, t+1, :] = eval_pred[:, t, :]

                    # Calculate metrics
                    eval_loss = self.criterion(eval_pred, target_eval)
                    obs_last_pos = obs_tensor_eval[:, -1:, 0:2]
                   
                    eval_mad, eval_fad = self.calculate_metrics(eval_pred, target_eval, obs_last_pos)
                    
                    eval_batch_metrics = {
                                'loss': eval_loss.item(),
                                'ade': eval_mad,
                                'fde': eval_fad
                            }
                    self.tracker.update(eval_batch_metrics, obs_tensor_eval.shape[0], phase='test')
                    
            self.tracker.compute_epoch_metrics(phase='test')
            # Print epoch metrics
            if not from_train:
                self.tracker.print_epoch_metrics(epoch=0, epochs=1, verbose=True)
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise        
        finally:
            # Restore original training mode
            super().train(training)
            
    def predict(self, obs_tensors):
        """
        Predict future absolute positions given a list of observed trajectories.
        
        Each observation in obs_tensors is a list (or array) of absolute positions [x, y]
        for one agent. The process is:
          1. For each trajectory, compute velocity differences (traj[1:] - traj[:-1]).
          2. Pad the velocity sequence at the beginning (if shorter than past_trajectory - 1).
          3. Stack the processed velocities into a batch tensor of shape 
             (B, past_trajectory - 1, 2).
          4. If normalization is enabled, normalize the velocities.
          5. Use these as the source (src) input to the Transformer.
          6. Autoregressively generate the target (tgt) sequence:
             - Initialize the decoder input with zeros (shape: (B, 1, 2)).
             - For each future time step, create a target mask and run the model to obtain 
               the next predicted velocity.
             - Append the new token to the decoder input.
          7. Denormalize the predicted velocities (if needed) and convert them to absolute positions
             by cumulatively summing starting from the last observed absolute position.
          8. Return the predictions as a list (one per agent) of NumPy arrays with shape 
             (future_trajectory, 2).
        """

        B = len(obs_tensors)
        desired_vel_len = self.past_trajectory - 1      
        processed_velocities = []
        last_positions = []
        for traj in obs_tensors:
            traj_tensor = torch.tensor(traj, dtype=torch.float32, device=self.device)
            last_positions.append(traj_tensor[-1, 0:2])
            if traj_tensor.shape[0] < 2:
                processed_velocities.append(torch.zeros((desired_vel_len, 2), device=self.device))
                continue
            
            obs_vel = traj_tensor[1:] - traj_tensor[:-1]  
            if obs_vel.shape[0] < desired_vel_len:
                pad_size = desired_vel_len - obs_vel.shape[0]
                pad = torch.zeros((pad_size, 2), dtype=torch.float32, device=self.device)
                obs_vel = torch.cat([pad, obs_vel], dim=0)
            elif obs_vel.shape[0] > desired_vel_len:
                obs_vel = obs_vel[-desired_vel_len:]
                
            processed_velocities.append(obs_vel)
        
        src = torch.stack(processed_velocities, dim=0)
        if self.normalized:
            src = (src - self.mean[2:]) / self.std[2:]
        
        future_len = self.future_trajectory
        decoder_input = torch.zeros(B, 1, self.output_features, device=self.device)
        pred_tokens = []
        
        with torch.no_grad():
            for t in range(future_len):
                current_tgt_len = decoder_input.size(1)
                tgt_mask = self._generate_square_mask(current_tgt_len, current_tgt_len, mask_type="tgt").to(self.device)
                out = self(src, decoder_input, tgt_mask=tgt_mask)
                next_token = out[:, -1, :] 
                pred_tokens.append(next_token)
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        
        pred_vel = torch.stack(pred_tokens, dim=1)
        
        if self.normalized:
            pred_vel = pred_vel * self.std[2:] + self.mean[2:]
        
        pred_positions_batch = []
        for i in range(B):
            last_pos = last_positions[i]  
            pred_vel_i = pred_vel[i]       
            pred_positions = torch.zeros(future_len, 2, device=self.device)
            pred_positions[0] = last_pos + pred_vel_i[0]
            for t in range(1, future_len):
                pred_positions[t] = pred_positions[t-1] + pred_vel_i[t]
            pred_positions_batch.append(pred_positions.cpu().numpy())
        
        return pred_positions_batch
    
    def calculate_metrics(self,pred: torch.Tensor, target: torch.Tensor, obs_last_pos: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate ADE and FDE for predictions
        Args:
            pred: predicted velocities [batch, seq_len, 2]
            target: target velocities [batch, seq_len, 2]
            obs_last_pos: last observed position [batch, 1, 2]
            mean: mean values for denormalization
            std: standard deviation values for denormalization
            device: computation device
        """
        if self.normalized:
            # Denormalize
            pred = pred * self.std[2:] + self.mean[2:]
            target = target * self.std[2:] + self.mean[2:]
        
        # Convert velocities to absolute positions through cumsum
        pred_pos = pred.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
        target_pos = target.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
        
        # Calculate metrics
        ade = calculate_ade(pred_pos, target_pos.tolist())
        fde = calculate_fde(pred_pos, target_pos.tolist())
        
        return ade, fde