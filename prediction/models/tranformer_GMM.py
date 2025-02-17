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
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from typing import Tuple, Dict, Optional, List, Union,Any
from dataclasses import dataclass, asdict
from evaluation.distance_metrics import calculate_ade,calculate_fde
from tqdm import tqdm
import os
import logging
import sys
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for the AttentionGMM model."""
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
    win_size: int = 3
    

    # GMM parameters
    n_gaussians: int = 6
    n_hidden: int = 32

    # Optimizer parameters
    lr_mul: float = 0.2
    n_warmup_steps: int = 3000 #2000 #3000 #3500
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-9

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
    def display_config(self, verbose: bool = False) -> None:
        """
        Pretty print the model configuration using logging.
        
        Args:
            verbose (bool): If True, logs additional information and formatting
        """
        logger = logging.getLogger('AttentionGMM')
        
        if verbose:
            logger.info("\n" + "="*50)
            logger.info("AttentionGMM Model Configuration")
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
            logger.info(f"Number of Gaussians: {self.n_gaussians}")
            logger.info(f"Hidden Size:         {self.n_hidden}")
            
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
                f"AttentionGMM Config: in_features={self.in_features}, "
                f"out_features={self.out_features}, num_heads={self.num_heads}, "
                f"embedding_size={self.embedding_size}, dropout={self.dropout}, "
                f"n_gaussians={self.n_gaussians}"
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
   
class AttentionGMM(nn.Module):
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
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__()
        # Create config object first
        self.config = config or ModelConfig(**kwargs)
        
        self._validate_config()
        self._init_device()
        self._init_model_params()
        self._init_layers()
        self._init_optimizer_params()
        
        self.tracker = MetricTracker()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.embedding_size % self.config.num_heads != 0:
            raise ValueError("Embedding size must be divisible by number of heads")
        if self.config.num_heads < 1:
            raise ValueError("Number of heads must be positive")
        if self.config.n_gaussians < 1:
            raise ValueError("Number of Gaussians must be positive")
        if self.config.n_hidden < 1:
            raise ValueError("Hidden size must be positive")
        if not 0 <= self.config.dropout <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if self.config.past_trajectory < 1 or self.config.future_trajectory < 1:
            raise ValueError("Trajectory lengths must be positive")
        
    def _init_device(self):
        """Initialize device configuration."""
        self.device = self.config.get_device()
        self.mean = self.config.mean.to(self.device)
        self.std = self.config.std.to(self.device)
        
    def _init_model_params(self):
        """Initialize model parameters."""
        self.num_heads = self.config.num_heads
        self.max_len = max(self.config.past_trajectory, self.config.future_trajectory)
        self.past_trajectory = self.config.past_trajectory
        self.future_trajectory = self.config.future_trajectory
        self.num_gaussians = self.config.n_gaussians
        self.hidden = self.config.n_hidden
        
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
            'fde': float('inf'),
            'best_ade': float('inf'),
            'best_fde': float('inf')
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
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # GMM layers
        self._init_gmm_layers()
    
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

    def _create_gmm_embedding(self):
        """Create a simple GMM embedding network."""
        return nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, int(self.hidden * 0.75)),
            nn.ELU(),
            nn.Linear(int(self.hidden * 0.75), self.hidden // 2),
            nn.ELU()
        )
    
    def _init_gmm_layers(self):
        """Initialize Gaussian Mixture Model layers."""
        # Create embedding networks
        self.embedding_sigma = self._create_gmm_embedding()
        self.embedding_mue = self._create_gmm_embedding()
        self.embedding_pi = self._create_gmm_embedding()
        
        # Create output heads
        self.pis_head = nn.Linear(self.hidden // 2, self.num_gaussians)
        self.sigma_head = nn.Linear(self.hidden // 2, self.num_gaussians * 2)
        self.mu_head = nn.Linear(self.hidden // 2, self.num_gaussians * 2)
    
    def _init_weights(self):
        """Initialize the model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
            # memory_mask=src_mask
        )
        

        # Compute embeddings
        sigma_embedded = self.embedding_sigma(decoder_output)
        mue_embedded = self.embedding_mue(decoder_output)
        pi_embedded = self.embedding_pi(decoder_output)  # <-- Apply embedding to pi

        
        # Mixture weights (apply softmax)
        pi = torch.softmax(self.pis_head(pi_embedded), dim=-1)
        
        # Compute Sigmas with softplus to ensure positivity
        sigma = nn.functional.softplus(self.sigma_head(sigma_embedded))
        sigma_x, sigma_y = sigma.chunk(2, dim=-1)
        
        # Compute Means
        mu = self.mu_head(mue_embedded)
        mu_x, mu_y = mu.chunk(2, dim=-1)
        
        return pi, sigma_x,sigma_y, mu_x ,mu_y #,decoder_output
    
    def train(
        self,
        train_dl: DataLoader,
        test_dl: DataLoader = None,
        epochs: int = 100,
        verbose: bool = True,
        save_path: str = 'results',
        save_model: bool = True,
        save_frequency: int = 50,
    ) -> Tuple[nn.Module, Dict]:
        """
        Train the model with metrics tracking and visualization.
        """
        # Setup logger
        logger = logging.getLogger('AttentionGMM')
        if not logger.handlers:
            logger = self.setup_logger(save_path=self.log_save_path)
        
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

        # Set up directory structure
        models_dir = os.path.join(save_path, 'pretrained_models')
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(models_dir, exist_ok=True)
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

                tgt = torch.zeros((target.shape[0], dec_seq_len, 2), dtype=torch.float32, device=self.device)


                # Generate masks
                tgt_mask = self._generate_square_mask(
                    dim_trg=dec_seq_len,
                    dim_src=updated_enq_length,
                    mask_type="tgt"
                ).to(self.device)
                

                # Forward pass
                optimizer.zero_grad()

                pi, sigma_x,sigma_y, mu_x , mu_y = self(input_train,tgt,tgt_mask = tgt_mask)
                mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
                sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

                
                # Calculate loss
                train_loss = self._mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,target,self.num_gaussians)
                
                # Backward pass
                train_loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                # print(f"Gradient Norm: {total_norm:.4f}")
                optimizer.step_and_update_lr()

                with torch.no_grad(): # to avoid data leakage during sampling
                    
                    highest_prob_pred, best_of_n_pred = self._sample_gmm_predictions(pi, sigmas, mus,target)
                    
                    obs_last_pos = obs_tensor[:, -1:, 0:2]

                    # using heighest probability values
                    mad, fad = self.calculate_metrics(
                        highest_prob_pred.detach(), target.detach(), obs_last_pos)
                    
                    # Best of n_predictions error
                    mad_best_n, fad_best_n = self.calculate_metrics(
                        best_of_n_pred.detach(), target.detach(), obs_last_pos)
                   
                    # Update metrics using tracker
                    batch_metrics = {
                        'loss': train_loss.item(),
                        'ade': mad,
                        'fde': fad,
                        'best_ade': mad_best_n,
                        'best_fde': fad_best_n
                    }
                    self.tracker.update(batch_metrics, obs_tensor.shape[0], phase='train')      
                    #Update progress bar
                    if verbose:
                        train_avgs = self.tracker.get_averages('train')
                        load_train.set_postfix({
                            'Loss': f"{train_avgs['loss']:.4f}",
                            'ADE': f"{train_avgs['ade']:.4f}",
                            'FDE': f"{train_avgs['fde']:.4f}",
                            'Best_ADE': f"{train_avgs['best_ade']:.4f}",
                            'Best_FDE': f"{train_avgs['best_fde']:.4f}"
                        })
                    
            # At end of epoch
            self.tracker.compute_epoch_metrics(phase='train')
            # Test evaluation
            if test_dl is not None:
                self.evaluate(test_dl,from_train=True)

            # Print epoch metrics
            self.tracker.print_epoch_metrics(epoch, epochs, verbose)

            # Check early stopping conditions
            phase = 'test' if test_dl else 'train'
            current_metrics = {
                'ade': self.tracker.get_averages(phase)['ade'],
                'fde': self.tracker.get_averages(phase)['fde'],
                'best_ade': self.tracker.get_averages(phase)['best_ade'],
                'best_fde': self.tracker.get_averages(phase)['best_fde']
            }
            
            should_stop, best_metrics = self.check_early_stopping(current_metrics, verbose)

            if save_model and (epoch + 1) % save_frequency == 0:
                model_state = {
                    'model_state_dict': self.state_dict(),  # Save directly
                    'optimizer_state_dict': optimizer._optimizer.state_dict(),
                    'training_history': self.tracker.history,
                    'best_metrics': self.tracker.best_metrics,
                    'train_mean': self.mean,
                    'train_std': self.std,
                    'num_gaussians': self.num_gaussians,
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
                # torch.save(model_state, os.path.join(models_dir, checkpoint_name))
                # Save the model
                checkpoint_name = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_Warm_{self.n_warmup_steps}_W_{self.config.win_size}.pth'
                os.makedirs(save_path, exist_ok=True)
                torch.save(model_state, os.path.join(models_dir, f"{checkpoint_name}"))
                logger.info(f"Saved checkpoint to: {save_path}")

            
            # Break if early stopping triggered
            if should_stop:
                logger.info("Early stopping triggered. Ending training.")
                break

            # Reset metrics for next epoch
            self.tracker.reset('train')
            self.tracker.reset('test')

        # Plot training history if verbose
        if verbose:
            self.plot_metrics(
                self.tracker.history['train_loss'],
                self.tracker.history['test_loss'],
                self.tracker.history['train_ade'],
                self.tracker.history['test_ade'],
                self.tracker.history['train_fde'],
                self.tracker.history['test_fde'],
                self.tracker.history['train_best_ade'],
                self.tracker.history['test_best_ade'],
                self.tracker.history['train_best_fde'],
                self.tracker.history['test_best_fde'],
                enc_seq_len,
                dec_seq_len
            )
            logger.info(f"Training plots saved to {metrics_dir}")

        return self, self.tracker.history

    @classmethod
    def load_model(self, ckpt_path: str):
        """
        Load a complete model with all necessary state.
        
        Args:
            ckpt_path (str): Path to checkpoint file
        """
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            
            # Load and move tensors to device
            self.mean = checkpoint['train_mean'].to(self.device)
            self.std = checkpoint['train_std'].to(self.device)
            self.num_gaussians = checkpoint['num_gaussians']
            
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
    
    def _sample_gmm_predictions(self, pi, sigma, mue, gt_normalized):
        """
        Returns both highest probability and best-of-N predictions
        
        Args:
            pi (torch.Tensor): Mixture weights (batch_size, seq_len, n_mixtures)
            sigma (torch.Tensor): Standard deviations
            mue (torch.Tensor): Means (batch_size, seq_len, n_mixtures, 2)
            gt_normalized: Normalized ground truth for best-of-N selection
            
        Returns:
            tuple: (highest_prob_pred, best_of_n_pred)
        """
        # 1. Get highest probability predictions
        max_indices = torch.argmax(pi, dim=2).unsqueeze(-1)
        highest_prob_pred = torch.gather(mue, dim=2, 
                                    index=max_indices.unsqueeze(dim=-1).repeat(1, 1, 1, 2))
        highest_prob_pred = highest_prob_pred.squeeze(dim=2)
        
        # 2. Get best-of-N predictions
        batch_size, seq_len, n_mixtures, _ = mue.shape
        best_of_n_pred = torch.zeros_like(highest_prob_pred)
        
    
        # Calculate errors for all mixtures at once
        expanded_gt = gt_normalized.unsqueeze(2).expand(-1, -1, n_mixtures, -1)
        errors = torch.norm(mue - expanded_gt, dim=-1)  # [batch_size, seq_len, n_mixtures]
        
        # Get best mixture for each timestep
        best_k = torch.argmin(errors, dim=-1)  # [batch_size, seq_len]
        
        # Create indices on the same device as input tensors
        batch_indices = torch.arange(batch_size, device=mue.device).view(-1, 1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=mue.device).view(1, -1).expand(batch_size, -1)
        
        # Gather predictions - all tensors are on the same device
        best_of_n_pred = mue[batch_indices, seq_indices, best_k]
        
        return highest_prob_pred, best_of_n_pred
    
    def _bivariate(self,pi,sigma_x,sigma_y, mu_x , mu_y,target_points):
        """
        Calculate bivariate Gaussian probability density.

        Args:
            pi (torch.Tensor): Mixture weights (batch_size, seq_len, n_mixtures)
            sigma_x (torch.Tensor): Standard deviations in x direction (batch_size, seq_len, n_mixtures)
            sigma_y (torch.Tensor): Standard deviations in y direction (batch_size, seq_len, n_mixtures)
            mu_x (torch.Tensor): Means in x direction (batch_size, seq_len, n_mixtures)
            mu_y (torch.Tensor): Means in y direction (batch_size, seq_len, n_mixtures)
            target_points (torch.Tensor): Target coordinates (batch_size, seq_len, 2) or (batch_size, 2)

        Returns:
            torch.Tensor: Log probability densities (batch_size, seq_len, n_mixtures)
        """
        # Validate input values
        if torch.isnan(sigma_x).any() or torch.isnan(sigma_y).any():
            raise ValueError("NaN values detected in sigma computation")    
        if torch.any(pi <= 0):
            raise ValueError("Mixture weights must be positive")

        # Validate shapes match
        expected_shape = pi.shape
        for tensor, name in [(sigma_x, 'sigma_x'), (sigma_y, 'sigma_y'), 
                            (mu_x, 'mu_x'), (mu_y, 'mu_y')]:
            if tensor.shape != expected_shape:
                raise ValueError(f"Shape mismatch: {name} has shape {tensor.shape}, "
                            f"expected {expected_shape}")

        # Extract x, y coordinates and add necessary dimensions
        if target_points.ndim == 3:
            x = target_points[:,:,0].unsqueeze(-1)
            y = target_points[:,:,1].unsqueeze(-1)
        elif target_points.ndim == 2:
            x = target_points[:,0].unsqueeze(1)
            y = target_points[:,1].unsqueeze(1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {target_points.ndim}D")

        # Calculate squared normalized distances
        norm_x = torch.square((x.expand_as(mu_x) - mu_x) * torch.reciprocal(sigma_x))
        norm_y = torch.square((y.expand_as(mu_y) - mu_y) * torch.reciprocal(sigma_y))

        # Calculate log probabilities
        exponent = -0.5 * (norm_x + norm_y)
        log_pi = torch.log(pi)
        log_normalization = -torch.log(2.0 * np.pi * sigma_x * sigma_y)
        
        return log_pi + log_normalization.expand_as(log_pi) + exponent
    def _mdn_loss_fn(self,pi, sigma_x,sigma_y, mu_x , mu_y,targets,n_mixtures):
        """
        Calculate the Mixture Density Network loss using LogSumExp trick for numerical stability.
        
        Args:
            pi (torch.Tensor): Mixture weights
            sigma_x (torch.Tensor): Standard deviations in x direction
            sigma_y (torch.Tensor): Standard deviations in y direction
            mu_x (torch.Tensor): Means in x direction
            mu_y (torch.Tensor): Means in y direction
            targets (torch.Tensor): Target points
            n_mixtures (int): Number of Gaussian mixtures
        
        Returns:
            torch.Tensor: Mean negative log likelihood loss
        """

        
        # Calculate log probabilities for each mixture component
        log_probs = self._bivariate(pi, sigma_x, sigma_y, mu_x, mu_y, targets)

        # Apply LogSumExp trick for numerical stability
        max_log_probs = torch.max(log_probs, dim=2, keepdim=True)[0]
        max_log_probs_repeated = max_log_probs.repeat(1, 1, n_mixtures)


        # Calculate stable log sum exp
        exp_term = torch.exp(log_probs - max_log_probs_repeated)
        epsilon = 0.00001 #torch.finfo(torch.float32).eps  # Use machine epsilon
        sum_exp = torch.sum(exp_term, dim=-1) + epsilon

        # Final loss calculation
        neg_log_likelihood = -(max_log_probs[:,:,0] + torch.log(sum_exp))
        
        # Check for numerical instability
        if torch.isnan(neg_log_likelihood).any():
            raise ValueError("NaN values detected in loss computation")

        return torch.mean(neg_log_likelihood)
    def check_early_stopping(self, current_metrics: dict, verbose: bool = True) -> Tuple[bool, dict]:
            """
            Check if training should stop based on the current metrics.
            
            Args:
                current_metrics (dict): Dictionary containing current metric values
                verbose (bool): Whether to print early stopping information
                
            Returns:
                Tuple[bool, dict]: (should_stop, best_metrics)
            """
            should_stop = True
            logger = logging.getLogger('AttentionGMM')
            
            # Check each metric for improvement
            for metric_name, current_value in current_metrics.items():
                if metric_name not in self.best_metrics:
                    continue
                    
                # Check if the current value is better than the best value
                if current_value < (self.best_metrics[metric_name] + self.config.early_stopping_delta):
                    self.best_metrics[metric_name] = current_value
                    should_stop = False
            
            # Update counter based on improvement
            if should_stop:
                self._early_stop_counter += 1
                if verbose and self._early_stop_counter > 0:
                    logger.info(f"\nNo improvement in metrics for {self._early_stop_counter} epochs.")
            else:
                self._early_stop_counter = 0
            
            # Check if we should stop training
            should_stop = self._early_stop_counter >= self.config.early_stopping_patience
            
        # Log early stopping information if triggered
            if should_stop and verbose:
                logger.info(f"\nEarly stopping triggered after {self._early_stop_counter} epochs without improvement")
                logger.info("Best metrics achieved:")
                for metric, value in self.best_metrics.items():
                    logger.info(f"Best {metric.upper()}: {value:.4f}")
            
            return should_stop, self.best_metrics.copy() 

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
    def evaluate(self, test_loader=None, ckpt_path=None, from_train=False):
        """
        Evaluate the model on test data
        Args:
            test_loader: DataLoader for test data
            ckpt_path: Path to checkpoint file
            from_train: Boolean indicating if called during training
        """
        logger = logging.getLogger('AttentionGMM')
    
        if test_loader is None:
            raise ValueError("test_loader cannot be None")
            
        # Store initial training mode
        training = self.training
        
        try:
            if not from_train:
                self.load_model(ckpt_path, device=self.device)(ckpt_path, self.device)                                                     
                        
           
            # Set evaluation mode :  Need to use nn.Module's train method for mode setting
            super().train(False)  
            self.tracker.test_available = True

            # logger.info(f"Starting evaluation on {len(test_loader)} batches")
            num_evaluated = 0
            
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
                    target_eval = (target_tensor_eval[:,:,2:4] - self.mean[2:])/self.std[2:]

                    tgt_eval = torch.zeros((target_eval.shape[0], dec_seq_len, 2), dtype=torch.float32, device=self.device)

                    tgt_mask = self._generate_square_mask(
                        dim_trg=dec_seq_len,
                        dim_src=updated_enq_length,
                        mask_type="tgt"
                    ).to(self.device)

                    pi_eval, sigma_x_eval,sigma_y_eval, mu_x_eval , mu_y_eval = self(input_eval,tgt_eval,tgt_mask = tgt_mask)
                    mus_eval = torch.cat((mu_x_eval.unsqueeze(-1),mu_y_eval.unsqueeze(-1)),-1)
                    sigmas_eval = torch.cat((sigma_x_eval.unsqueeze(-1),sigma_y_eval.unsqueeze(-1)),-1)

                    # highest_prob_pred and best of n prediction
                    highest_prob_pred, best_of_n_pred = self._sample_gmm_predictions(pi_eval, sigmas_eval, mus_eval,target_eval)
                    
                    # Calculate metrics
                    eval_loss = self._mdn_loss_fn(pi_eval, sigma_x_eval,sigma_y_eval, mu_x_eval , mu_y_eval,target_eval,self.num_gaussians)
                    eval_obs_last_pos = obs_tensor_eval[:, -1:, 0:2]

                    eval_ade, eval_fde = self.calculate_metrics(highest_prob_pred, target_eval, eval_obs_last_pos)

                    eval_ade_best_n, eval_fde_best_n = self.calculate_metrics(best_of_n_pred, target_eval, eval_obs_last_pos)
                    
                    batch_metrics = {
                                'loss': eval_loss.item(),
                                'ade': eval_ade,
                                'fde': eval_fde,
                                'best_ade': eval_ade_best_n,
                                'best_fde': eval_fde_best_n
                            }
                    num_evaluated += obs_tensor_eval.shape[0]
                    self.tracker.update(batch_metrics, obs_tensor_eval.shape[0], phase='test')
                    
                    # if hasattr(torch.cuda, 'empty_cache'):
                    #     torch.cuda.empty_cache()

            # logger.info(f"Completed evaluation of {num_evaluated} samples")
            self.tracker.compute_epoch_metrics(phase='test')
            # Print epoch metrics
            if not from_train:
                self.tracker.print_epoch_metrics(epoch=1, epochs=1, verbose=True)
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise        
        finally:
            # Restore original training mode
            super().train(training)
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
    
    def setup_logger(self,name: str = 'AttentionGMM', save_path: str = None, level=logging.INFO):
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
            log_path = Path(save_path) / f'training_metrics_model_{self.past_trajectory}_{self.future_trajectory}_training_{self.n_warmup_steps}_W_{self.config.win_size}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path),mode='w')
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def plot_metrics(self,
        train_losses: List[float],
        test_losses: List[float],
        train_ades: List[float],
        test_ades: List[float],
        train_fdes: List[float],
        test_fdes: List[float],
        train_best_ades: List[float],
        test_best_ades: List[float],
        train_best_fdes: List[float],
        test_best_fdes: List[float],
        enc_seq_len: int,
        dec_seq_len: int,
    ) -> None:
        """Plot training metrics including best-of-N predictions.
        
        Args:
            train_losses: Training loss values
            test_losses: Test loss values
            train_ades: Training ADE values
            test_ades: Test ADE values
            train_fdes: Training FDE values
            test_fdes: Test FDE values
            train_best_ades: Training Best-of-N ADE values
            test_best_ades: Test Best-of-N ADE values
            train_best_fdes: Training Best-of-N FDE values
            test_best_fdes: Test Best-of-N FDE values
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
        ax2.plot(train_best_ades, label='Train Best ADE', color='blue', linestyle='--')
        ax2.plot(test_best_ades, label='Test Best ADE', color='orange', linestyle='--')
        ax2.set_title('Average Displacement Error (ADE)', pad=20)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('ADE Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # FDE plot
        ax3.plot(train_fdes, label='Train FDE', color='blue')
        ax3.plot(test_fdes, label='Test FDE', color='orange')
        ax3.plot(train_best_fdes, label='Train Best FDE', color='blue', linestyle='--')
        ax3.plot(test_best_fdes, label='Test Best FDE', color='orange', linestyle='--')
        ax3.set_title('Final Displacement Error (FDE)', pad=20)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('FDE Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(self.log_save_path, exist_ok=True)
        save_file = os.path.join(self.log_save_path, f'training_metrics_model_{enc_seq_len}_{dec_seq_len}_W_{self.config.win_size}.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

class Linear_Embeddings(nn.Module):
    def __init__(self, input_features,d_model):
        super(Linear_Embeddings, self).__init__()
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
            'train_fde': [], 'test_fde': [],
            'train_best_ade': [], 'test_best_ade': [],
            'train_best_fde': [], 'test_best_fde': []
        }

        self.best_metrics = {'ade': float('inf'), 'epoch': 0}

    def _init_metric_dict(self):
        """Helper to initialize metrics dictionary."""
        return {key: {'value': 0, 'count': 0} for key in ['loss', 'ade', 'fde', 'best_ade', 'best_fde']}
    
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
        self.history[f'{phase}_best_ade'].append(epoch_metrics['best_ade'])
        self.history[f'{phase}_best_fde'].append(epoch_metrics['best_fde'])

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
            'fde': self.history[f'{phase}_fde'][-1],
            'best_ade': self.history[f'{phase}_best_ade'][-1],
            'best_fde': self.history[f'{phase}_best_fde'][-1]
        }

    def get_previous_epoch_metrics(self, phase='train'):
        """Get previous epoch metrics."""
        if len(self.history[f'{phase}_loss']) < 2:  # need at least 2 epochs
            return None
            
        return {
            'loss': self.history[f'{phase}_loss'][-2],
            'ade': self.history[f'{phase}_ade'][-2],
            'fde': self.history[f'{phase}_fde'][-2],
            'best_ade': self.history[f'{phase}_best_ade'][-2],
            'best_fde': self.history[f'{phase}_best_fde'][-2]
        }
    def print_epoch_metrics(self, epoch, epochs, verbose=True):
        """Print epoch metrics including best-of-N results in a side-by-side format."""
        if not verbose:
            return

        logger = logging.getLogger('AttentionGMM')
        
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
        for metric, name in [('loss', 'Loss'), ('ade', 'ADE'), ('fde', 'FDE'),
                            ('best_ade', 'Best ADE'), ('best_fde', 'Best FDE')]:
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