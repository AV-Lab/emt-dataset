#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:08:42 2024

@author: nadya and somehow murad contributed
"""

import os
import cv2
import sys
import argparse
import numpy as np
import json    
from torch.utils.data import DataLoader
import torch.cuda as cuda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_meta_data
from utils import generate_prediction_settings
from utils import generate_intention_settings

from evaluation.distance_metrics import calculate_ade, calculate_fde
from dataloaders.seq_loader import SeqDataset

from models.rnn import RNNPredictor
from models.transformer_model import AttentionEMT ,Attention_EMT
import torch
import torch.nn as nn

from train import train_attn,ScheduledOptim
import numpy as np
from utils import set_seeds


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run predictors')
    p.add_argument('past_trajectory',type=int, help='Past Trajectory')
    p.add_argument('future_trajectory',type=int, help='Prediction Horizon')
    p.add_argument('epochs', type=str, default=50, help='Num of training epochs')
    p.add_argument('--window_size', default=1, type=int, help='Sliding window')
    p.add_argument('--predictor', type=str, choices=['lstm', 'gnn', 'transformer'], default='transformer',help='Predictor type')
    p.add_argument('--setting', type=str, default='train',choices=['train', 'evaluate'], help='Execution mode (train or evaluate)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file, required if mode is evaluate')
    p.add_argument('--annotations_path', type=str, default="data/annotations", help='If annotations are placed in a location different from recomended')
    p.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--device', type=str, default='cuda', help='device to run the model',choices=['cuda', 'cpu'])

    args = p.parse_args()
   

    if args.setting == "evaluate" and not args.checkpoint:
        print("No checkpoint provided")
        exit
    elif args.setting == "train" and not args.checkpoint:
        args.checkpoint = f'transformer_P_{args.past_trajectory}_F_{args.future_trajectory}_W_{args.window_size}.pth'
    if args.device == "cuda" and not cuda.is_available():
        args.device = "cpu"
        print("Could not find GPU. Using CPU instead!")

    

    # Print all arguments
    print("\nRunning with the following parameters:")
    for arg in vars(args):
        print(f"{arg:20s}: {getattr(args, arg)}")
    
    
    # set seed for deterministic training
    set_seeds(seed = 42)



    # Generate setting
    ann_path = "../data/annotations" if not args.annotations_path else args.annotations_path
    prd_ann_path = ann_path + "/prediction_annotations"
    annotations = [prd_ann_path + '/' + f for f in os.listdir(prd_ann_path)]
    splits = load_meta_data(ann_path + "/metadata.txt")
    data_folder = generate_prediction_settings(args.past_trajectory, args.future_trajectory, splits, annotations, args.window_size)
    
    
    # # Create DataLoaders : Transformer and GMM models use relative position rather than absolute position 
    if args.predictor=='transformer' or args.predictor=='gmm': 
        include_velocity = True
    else:
        include_velocity = False
        
    tain_dataset = SeqDataset(data_folder,"train",include_velocity)
    train_dataloader = DataLoader(tain_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
    test_datatset = SeqDataset(data_folder, "test",include_velocity)
    test_dataloader = DataLoader(test_datatset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)


    # get mean and standard deviation
    train_mean,train_std = tain_dataset.mean,tain_dataset.std
   


    # Train Predictor
    if args.predictor=='transformer':
        # # Initialize model
        transformer = AttentionEMT(max_length=max(args.past_trajectory, args.future_trajectory)).to(args.device)

        transformer = AttentionEMT(
            in_features=2,
            out_features=2,
            num_heads=2,
            num_encoder_layers=3,
            num_decoder_layers=3,
            embedding_size=128,
            dropout=0.1,
            max_length=max(args.past_trajectory, args.future_trajectory),
            batch_first=True,
            actn="gelu",
        ).to(args.device)
        
        # Train the model
        model, history = transformer.train_model(args=args,train_dl=train_dataloader,test_dl=test_dataloader,epochs=50,mean=train_mean,std=train_std,verbose=True)

        # # Train the model
        # model, history = transformer.train_model(
        #     args=args,
        #     train_dl=train_dataloader,
        #     test_dl=test_dataloader,
        #     epochs=int(args.epochs) if hasattr(args, 'epochs') else 2,
        #     mean=train_mean,
        #     std=train_std,
        #     verbose=True
        # )
        # Initialize model
        # transformer_model = Attention_EMT(
        #     in_features = 2,
        #     out_features = 2,
        #     num_heads = 2,
        #     num_encoder_layers = 3,
        #     num_decoder_layers = 3,
        #     embedding_size = 128,
        #     dropout = 0.1,
        #     max_length = max(args.past_trajectory, args.future_trajectory),
        #     batch_first=True,
        #     actn="gelu",
        #     device = args.device
        # ).to(args.device)  # Move model to device
        

        # # Setup optimizer parameters
        # args.lr_mul = 0.1
        # args.d_model = 128  # Should match embedding_size
        # args.n_warmup_steps = 3500
        
        # # Define the optimizer
        # optimizer = ScheduledOptim(
        #     torch.optim.Adam(transformer_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        #     args.lr_mul, 
        #     args.d_model, 
        #     args.n_warmup_steps
        # )

        # # Train model
        # trained_model, history = train_attn(
        #     args=args,
        #     train_dl=train_dataloader,
        #     test_dl=test_dataloader,
        #     model=transformer_model,  # Pass the model
        #     optim=optimizer,         # Pass the optimizer
        #     mean=train_mean,
        #     std=train_std,
        #     epochs=int(args.epochs) if hasattr(args, 'epochs') else 2  # Add epochs parameter
        # )

        # # Save model
        # # First move model to CPU
        # model_cpu = trained_model.to('cpu')

        # # Create model state dictionary excluding device info
        # model_state = {
        #     'model_state_dict': model_cpu.state_dict(),
        #     # 'optimizer_state_dict': optimizer.state_dict(),
        #     'training_history': history,
        #     'train_mean': train_mean,
        #     'train_std': train_std,
        #     'model_config': {
        #         'in_features': 2,
        #         'out_features': 2,
        #         'num_heads': 2,
        #         'num_encoder_layers': 3,
        #         'num_decoder_layers': 3,
        #         'embedding_size': 128,
        #         'dropout': 0.1,
        #         'max_length': max(args.past_trajectory, args.future_trajectory),
        #         'batch_first': True,
        #         'actn': "gelu"
        #     }
        # }

        # # Save the model
        # torch.save(model_state,'prediction/pre_trained/' +str(args.batch_size) + '_'+ args.checkpoint)

    
    # # Train Predictor 
    # predictor = RNNPredictor(args.past_trajectory, args.future_trajectory)
    # predictor.train(train_loader)
    # predictor.evaluate(test_loader)
    
    
    # Evaluate













































'''
A wrapper class for scheduled optimizer 
source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
'''
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

@dataclass
class ModelConfig:
    """Configuration for the AttentionEMT model."""
    # Model architecture parameters
    in_features: int = 2
    out_features: int = 2
    num_heads: int = 2
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    embedding_size: int = 128
    dropout: float = 0.2
    max_length: int = 12
    batch_first: bool = True
    actn: str = "gelu"
    # Optimizer parameters
    lr_mul: float = 0.1
    n_warmup_steps: int = 3500
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-9
    
    def to_dict(self) -> Dict[str, any]:
        """Convert config to dictionary."""
        return asdict(self)
    
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
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        super().__init__()

        # Use provided config or create from kwargs
        self.config = config or ModelConfig(**kwargs)
        set_seeds(42)
        
        # Store model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = self.config.num_heads
        self.max_len = self.config.max_length
        self.d_model = self.config.embedding_size
        self.input_features = self.config.in_features
        self.output_features = self.config.out_features

        # Store optimizer parameters
        self.lr_mul = self.config.lr_mul 
        self.n_warmup_steps = self.config.n_warmup_steps
        self.optimizer_betas = self.config.optimizer_betas
        self.optimizer_eps = self.config.optimizer_eps
        
        # Define dropout rates
        self.dropout_encoder = self.config.dropout
        self.dropout_decoder = self.config.dropout
        self.dropout_pos_enc = 0.0
        
        # Set feedforward dimensions (4x larger than d_model as per original paper)
        self.dim_feedforward = 4 * self.d_model
        
        # Initialize embeddings and positional encoding
        self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model)
        self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)
        
        self.positional_encoding = PositionalEncoding(
            d_model = self.d_model,
            dropout = self.dropout_pos_enc,
            max_len = self.max_len,
            batch_first = self.config.batch_first
        )
        
        # Initialize encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout_encoder,
            batch_first = self.config.batch_first,
            activation = self.config.actn 
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = self.config.num_encoder_layers
        )
        
        # Initialize decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.d_model,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout_decoder,
            batch_first = self.config.batch_first,
            activation = self.config.actn 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = self.config.num_encoder_layers
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(self.d_model, self.output_features)
        
        # Initialize weights using Xavier uniform initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None) -> 'AttentionEMT':
        """
        Load a saved model from disk.
        
        Args:
            path (str): Path to the saved model file
            device (torch.device, optional): Device to load the model to
            
        Returns:
            AttentionEMT: Loaded model instance
        """
        # Load the saved state
        state = torch.load(path, map_location='cpu')
        
        # Create model config from saved state
        config = ModelConfig(**state['model_config'])
        
        # Create new model instance
        model = cls(config=config)
        
        # Load state dict
        model.load_state_dict(state['model_state_dict'])
        
        # Move to specified device or use default
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model

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
    

    def train_model(
        self,
        args,
        train_dl: DataLoader,
        test_dl: DataLoader,
        epochs: int = 50,
        mean: torch.tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]),
        std: torch.tensor = torch.tensor([1.0, 1.0, 1.0, 1.0]),
        verbose: bool = True,
        save_path: str = 'prediction/results',
        save_model: bool = True,
        save_frequency: int = 10,
        # checkpoint_name: str = 'model.pt'
    ) -> Tuple[nn.Module, Dict]:
        """
        Train the model with metrics tracking and visualization.
        """
        # Setup optimizer with model's configuration
        optimizer = self.configure_optimizer(
            lr_mul=self.lr_mul,
            n_warmup_steps=self.n_warmup_steps,
            optimizer_betas=self.optimizer_betas,
            optimizer_eps=self.optimizer_eps
        )
        
        if verbose:
            print('Training Settings:')
            print(f"Train batch size: {args.batch_size}")
            print(f"Epochs: {epochs}")

        mean = mean.to(args.device)
        std = std.to(args.device)
        criterion = nn.MSELoss()
        
        # Initialize tracking
        train_losses, test_losses = [], []
        train_ades, test_ades = [], []
        train_fdes, test_fdes = [], []

        # Set up directory structure
        models_dir = os.path.join(save_path, 'pretrained_models')
        metrics_dir = os.path.join(save_path, 'metrics')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_ade = 0
            epoch_fde = 0
            self.train()  # Set train mode again for safety

            # Training loop with progress bar
            load_train = tqdm(train_dl, desc=f"Epoch: {epoch+1}/{epochs}") if verbose else train_dl

            for id_b, batch in enumerate(load_train):
                # Prepare input data
                obs_tensor, target_tensor = batch
                batch_size, enc_seq_len, feat_dim = obs_tensor.shape
                dec_seq_len = target_tensor.shape[1]
                
                # Move to device and normalize
                obs_tensor = obs_tensor.to(args.device)
                target_tensor = target_tensor.to(args.device)

                input = (obs_tensor[:,1:,2:4] - mean[2:])/std[2:]
                updated_enq_length = input.shape[1]
                target = (target_tensor[:,:,2:4] - mean[2:])/std[2:]

                # Prepare target input (teacher forcing)
                tgt = torch.zeros_like(target).to(args.device)
                tgt[:, 1:, :] = target[:, :-1, :]

                # Generate masks
                tgt_mask = self._generate_square_mask(
                    dim_trg=dec_seq_len,
                    dim_src=updated_enq_length,
                    mask_type="tgt"
                ).to(args.device)
                
                memory_mask = self._generate_square_mask(
                    dim_trg=dec_seq_len,
                    dim_src=updated_enq_length,
                    mask_type="memory"
                ).to(args.device)

                # Forward pass
                optimizer.zero_grad()
                pred = self(input, tgt, tgt_mask=tgt_mask)
                
                # Calculate loss
                train_loss = criterion(pred, target)
                
                # Backward pass
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step_and_update_lr()

                # Calculate metrics
                obs_last_pos = obs_tensor[:, -1:, 0:2]
                mad, fad = self.calculate_metrics(
                    pred.detach(), target, obs_last_pos,
                    True, mean, std, args.device
                )
                
                # Update epoch metrics
                epoch_loss += train_loss.item()
                epoch_ade += mad
                epoch_fde += fad
                
                # Update progress bar
                if verbose:
                    load_train.set_postfix({
                        'loss': f"{train_loss.item():.4f}",
                        'ADE': f"{mad:.4f}",
                        'FDE': f"{fad:.4f}"
                    })

            # Calculate average training metrics
            avg_train_loss = epoch_loss / len(train_dl)
            avg_train_ade = epoch_ade / len(train_dl)
            avg_train_fde = epoch_fde / len(train_dl)
            
            # Test evaluation
            self.eval()
            test_loss = 0
            test_ade = 0
            test_fde = 0
            
            with torch.no_grad():
                for batch in test_dl:
                    obs_tensor, target_tensor = batch
                    obs_tensor = obs_tensor.to(args.device)
                    target_tensor = target_tensor.to(args.device)

                    input = (obs_tensor[:,1:,2:4] - mean[2:])/std[2:]
                    updated_enq_length = input.shape[1]
                    target = (target_tensor[:,:,2:4] - mean[2:])/std[2:]

                    tgt = torch.zeros_like(target).to(args.device)
                    tgt[:, 1:, :] = target[:, :-1, :]

                    tgt_mask = self._generate_square_mask(
                        dim_trg=dec_seq_len,
                        dim_src=updated_enq_length,
                        mask_type="tgt"
                    ).to(args.device)

                    pred = self(input, tgt, tgt_mask=tgt_mask)
                    
                    # Calculate metrics
                    loss = criterion(pred, target)
                    obs_last_pos = obs_tensor[:, -1:, 0:2]
                    mad, fad = self.calculate_metrics(
                        pred, target, obs_last_pos,
                        True, mean, std, args.device
                    )
                    
                    test_loss += loss.item()
                    test_ade += mad
                    test_fde += fad

            # Average test metrics
            avg_test_loss = test_loss / len(test_dl)
            avg_test_ade = test_ade / len(test_dl)
            avg_test_fde = test_fde / len(test_dl)

            # Save metrics
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            train_ades.append(avg_train_ade)
            test_ades.append(avg_test_ade)
            train_fdes.append(avg_train_fde)
            test_fdes.append(avg_test_fde)

            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train - Loss: {avg_train_loss:.4f}, ADE: {avg_train_ade:.4f}, FDE: {avg_train_fde:.4f}")
                print(f"Test  - Loss: {avg_test_loss:.4f}, ADE: {avg_test_ade:.4f}, FDE: {avg_test_fde:.4f}")

            # Save model if requested
            if save_model and (epoch + 1) % save_frequency == 0:
                # Move model to CPU
                model_cpu = self.to('cpu')

                # Create model state dictionary
                model_state = {
                    'model_state_dict': model_cpu.state_dict(),
                    'optimizer_state_dict': optimizer._optimizer.state_dict(),
                    'training_history': {
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                        'train_ades': train_ades,
                        'test_ades': test_ades,
                        'train_fdes': train_fdes,
                        'test_fdes': test_fdes
                    },
                    'train_mean': mean,
                    'train_std': std,
                    'model_config': {
                        'in_features': self.input_features,
                        'out_features': self.output_features,
                        'num_heads': self.num_heads,
                        'num_encoder_layers': self.config.num_encoder_layers,
                        'num_decoder_layers': self.config.num_decoder_layers,
                        'embedding_size': self.d_model,
                        'dropout': self.dropout_encoder,
                        'max_length': self.max_len,
                        'batch_first': True,
                        'actn': self.config.actn
                    }
                }

                # Save the model
                checkpoint_name = args.checkpoint
                os.makedirs(save_path, exist_ok=True)
                torch.save(model_state, os.path.join(save_path, f"{checkpoint_name}"))
                print("saving checkpoint at : ", os.path.join(save_path, f"{checkpoint_name}"))
                
                # Move model back to device
                self.to(args.device)
        # Plot training history
        if verbose:
            self.plot_metrics(
                train_losses, test_losses,
                train_ades, test_ades,
                train_fdes, test_fdes,
                enc_seq_len, dec_seq_len,batch_size,
                save_path=metrics_dir
            )

        # Return model and history
        history = {
            'train_losses': train_losses, 'test_losses': test_losses,
            'train_ades': train_ades, 'test_ades': test_ades,
            'train_fdes': train_fdes, 'test_fdes': test_fdes
        }
        
        return self, history

    @staticmethod
    def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, obs_last_pos: torch.Tensor, 
                        normalized: bool, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> Tuple[float, float]:
        """
        Calculate ADE and FDE for predictions
        Args:
            pred: predicted velocities [batch, seq_len, 2]
            target: target velocities [batch, seq_len, 2]
            obs_last_pos: last observed position [batch, 1, 2]
            normalized: whether predictions are normalized
            mean: mean values for denormalization
            std: standard deviation values for denormalization
            device: computation device
        """
        if normalized:
            # Denormalize
            pred = pred * std[2:].to(device) + mean[2:].to(device)
            target = target * std[2:].to(device) + mean[2:].to(device)
        
        # Convert velocities to absolute positions through cumsum
        pred_pos = pred.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
        target_pos = target.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
        
        # Calculate metrics
        ade = calculate_ade(pred_pos, target_pos.tolist())
        fde = calculate_fde(pred_pos, target_pos.tolist())
        
        return ade, fde

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

    @staticmethod
    def plot_metrics(
        train_losses: List[float],
        test_losses: List[float],
        train_ades: List[float],
        test_ades: List[float],
        train_fdes: List[float],
        test_fdes: List[float],
        enc_seq_len: int,
        dec_seq_len: int,
        batch_size: int,
        save_path: str =  f'prediction/pre_trained/metrics/training_metrics'
    ) -> None:
        """Plot training metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(test_losses, label='Test Loss')
        ax1.set_title('Loss')
        ax1.legend()
        
        # ADE plot
        ax2.plot(train_ades, label='Train ADE')
        ax2.plot(test_ades, label='Test ADE')
        ax2.set_title('ADE')
        ax2.legend()
        
        # FDE plot
        ax3.plot(train_fdes, label='Train FDE')
        ax3.plot(test_fdes, label='Test FDE')
        ax3.set_title('FDE')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_metrics_batch-size_{batch_size}_model_{enc_seq_len}_{dec_seq_len}.png')
        plt.close()
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

    
