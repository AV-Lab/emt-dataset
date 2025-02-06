import torch 
from torch import Tensor
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
import math
from torch.autograd import Variable # storing data while learning
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, asdict
from evaluation.distance_metrics import calculate_ade,calculate_fde
from utils import set_seeds
import os

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

class Attention_EMT(nn.Module):
    def __init__(self,
        in_features = 2,
        out_features = 2,
        num_heads = 2,
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        embedding_size = 128,
        dropout = 0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
            super(Attention_EMT, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = 0.0
            self.ndim = 2

            
        
            
            #Following the original transformer paper's design where dim_feedforward is typically 4x larger than d_model
            self.dim_feedforward_encoder = 4*self.d_model
            self.dim_feedforward_decoder = 4*self.d_model


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model
            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )
            
            # Output layer (to map decoder output to the target shape)
            self.output_layer = nn.Linear(self.d_model, self.output_features)  

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        
        # Move inputs to device at the start
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self.device)

        # Embedding 
        encoder_embed = self.encoder_input_layer(src)
        encoder_embed = self.positional_encoding_layer(encoder_embed) #,src_mask=src_mask # Pass src_mask if used

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt)
        decoder_output = self.positional_encoding_layer(decoder_embed) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask
            # memory_mask=src_mask
            )
        
        # Output projection shape: [batch_size, target seq len, dim_val]
        output = self.output_layer(decoder_output)

        return  output



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
            num_layers = self.config.num_decoder_layers
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




class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # lut => lookup table
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class Linear_Embeddings(nn.Module):
    def __init__(self, input_features,d_model):
        super(Linear_Embeddings, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(input_features, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class AttentionMDN_Old(nn.Module):
    def __init__(self,
        device,
        num_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        n_gaussians = 5,
        n_hidden = 10,
        max_length = 8,
        batch_first = True
        ):
            super(AttentionMDN_Old, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = num_features
            self.output_features = num_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= 512 # selected
            self.dropout_encoder = 0.2
            self.dropout_decoder = 0.2
            self.dropout_pos_enc = 0.1
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.flatten = nn.Flatten(start_dim=1)
            self.ndim = 2
            self.relu = nn.ReLU()
            self.mdn_input_shape = self.output_features * self.max_len # input features*seq_length

            
            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )
            #self.pos_encoder = self.positional_encoding(self.max_len, self.d_model).to(device)

            # Creating the  linear layers needed for the model
            self.encoder_input_layer = nn.Linear(
                in_features= self.input_features , 
                out_features= self.d_model 
                )

            self.decoder_input_layer = nn.Linear(
                in_features = self.output_features,
                out_features = self.d_model 
                )  

            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first
                )


            # Stack the encoder layers in nn.TransformerDecoder
            # It seems the option of passing a normalization instance is redundant
            # in my case, because nn.TransformerEncoderLayer per default normalizes
            # after each sub-layer
            # (https://github.com/pytorch/pytorch/issues/24930).
            self.encoder = nn.TransformerEncoder(
                encoder_layer = encoder_layer,
                num_layers = self.num_encoder_layers, 
                norm=None
                )
            # Create the decoder layer
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first
            )

            # Stack the decoder layers in nn.TransformerDecoder
            # It seems the option of passing a normalization instance is redundant
            # in my case, because nn.TransformerDecoderLayer per default normalizes
            # after each sub-layer
            # (https://github.com/pytorch/pytorch/issues/24930).
            self.decoder = nn.TransformerDecoder(
                decoder_layer = decoder_layer,
                num_layers=self.num_decoder_layers, 
                norm=None
                )

            self.out = nn.Linear(self.d_model, self.output_features)

            # Mixed Density Network 

            # if the input of linear is associated with out_features, then the number of nodes will be 2
            # This is not useful coz it learns relation of x and y only
            # instead relation of sequence should also be learned (batch_size,-1) the -1 indicates feature_size*seq_length

            self.z_h = nn.Sequential(
            nn.Linear(self.mdn_input_shape,self.hidden),
            #nn.Sigmoid()
            nn.LeakyReLU(0.1)
            )

            self.z_pi_old = nn.Linear(self.hidden, self.gaussians).to(device)
            self.z_sigma_old = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)
            self.z_mu_old = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)  

            self.z_pi = nn.Linear(self.hidden,self.gaussians).to(device)
            self.z_sigma = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)
            self.z_mu = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)

            #self.z_sigma = nn.Linear(self.hidden, self.out_length*self.gaussians*self.ndim).to(device) 
    

    # def positional_encoding(self, max_len, d_model):
    #     pos = torch.arange(0, max_len).unsqueeze(1)
    #     i = torch.arange(0, d_model, 2).float()
    #     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    #     pe = pos * div_term.unsqueeze(0)
    #     pe[:, 0::2] = torch.sin(pe[:, 0::2])
    #     pe[:, 1::2] = torch.cos(pe[:, 1::2])
    #     return pe.unsqueeze(0)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        src = self.encoder_input_layer(src).to(device)
        # print("Starting ...")

        src = self.positional_encoding_layer(src).to(device) 


        # print("Source and Target positional encoding Done!")
        # print("Transformer Started ...",src.shape)

        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
        src=src
        ).to(device)

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt).to(device)

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        decoder_output = self.out(decoder_output).to(device)



        # transformer_out = self.transformer(src, tgt).to(device)
        #print("Transformer  Finished!")

        #print("decoder_output shape: ",decoder_output.shape)

        #x = 
        # x = self.flatten(decoder_output).to(device)
        #print("x shape: ",x.shape)
        z_h = self.z_h(decoder_output).to(device)
        #print("z_h shape: ",z_h.shape)

        # Calculate PI
        pi = self.z_pi(z_h).to(device)
        # print("pi before shape: ",pi.shape)
        # Reshape Pi before softmax 
        # pi = pi.view(-1,self.out_length,self.gaussians)
        # print("pi honest shape: ",pi.shape)
        # Softmax for probablistic output
        pi = nn.functional.softmax(pi, -1)
        # print("pi final shape: ",pi.shape)
        
        # Calculate Sigma and Mu
        sigma = torch.exp(self.z_sigma(z_h)).to(device)
        mu = self.z_mu(z_h).to(device)
    

        sigma_x = sigma[:,:,:self.gaussians]
        sigma_y = sigma[:,:,self.gaussians:]
        mu_x = mu[:,:,:self.gaussians]
        mu_y = mu[:,:,self.gaussians:]

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output

class Transformer_MDN(nn.Module):
    def __init__(self,
        device,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Transformer_MDN, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = 0.2
            self.dropout_decoder = 0.2
            self.dropout_pos_enc = 0.1
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            #self.pos_encoder = self.positional_encoding(self.max_len, self.d_model).to(device)



            # Creating the  linear layers needed for the model
            # self.encoder_input_layer = nn.Linear(
            #     in_features= self.input_features , 
            #     out_features= self.d_model 
            #     )
            
            # self.encoder_input_layer = nn.Linear(
            #     in_features= self.input_features , 
            #     out_features= self.d_model 
            #     )

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   

            # # Layer Normalization
            # self.norm_layer = nn.LayerNorm(self.d_model)

            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )

            self.out = nn.Linear(self.d_model, self.output_features)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        decoder_output = self.out(decoder_output).to(device)

        result = torch.Tensor(decoder_output)

        return  result

class Attention_GMM(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        dropout=0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_GMM, self).__init__()
            # self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = dropout
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
            self.mdn_weight =torch.tensor(0.5)
            # self.mdn_weight = nn.Parameter(torch.tensor(0.5),requires_grad=True)
            #self.mdn_weight = nn.Parameter(torch.tensor([0.5]),requires_grad=True).to(device)
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )


            self.embedding_sigma = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )
            self.embedding_mue = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )

            self.pis = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU(),
            nn.Linear(self.hidden//4,self.gaussians)
            #nn.Softmax()
            )

            
            self.hidden_hid = self.hidden//4
            # self.pis = nn.Linear(self.hidden_hid,self.gaussians).to(device)
            self.sigma_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.sigma_y = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_y = nn.Linear(self.hidden_hid, self.gaussians)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        #mdn_embeded = self.embedding_mdn(decoder_output).to(device)
        sigmax_embeded = self.embedding_sigma(decoder_output).to(device)
        sigmay_embeded = self.embedding_sigma(decoder_output).to(device)
        muex_embeded = self.embedding_mue(decoder_output).to(device)
        muey_embeded = self.embedding_mue(decoder_output).to(device)

        # Calculate PI
        pi = self.pis(decoder_output).to(device)
        pi = nn.functional.softmax(pi, -1)

        # Calculate Sigmas
        sigma_x = torch.Tensor(torch.exp(self.sigma_x(sigmax_embeded))).to(device)
        sigma_y = torch.Tensor(torch.exp(self.sigma_x(sigmay_embeded))).to(device)

        mu_x = torch.Tensor(self.mu_x(muex_embeded)).to(device)
        mu_y = torch.Tensor(self.mu_y(muey_embeded)).to(device)
       
        #result = torch.Tensor(decoder_output)

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output
    
class Attention_GMM_Encoder(nn.Module):
    def __init__(self,
        device,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        dropout=0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_GMM_Encoder, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = dropout
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
            #self.mdn_weight = nn.Parameter(torch.tensor([.5]))
            
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )

            #self.out = nn.Linear(self.d_model, self.output_features)
            #self.hidden_hid = self.hidden//2

            self.embedding_sigma = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.GELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.self.hidden//2),
            nn.GELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.GELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )
            self.embedding_mue = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.GELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.self.hidden//2),
            nn.GELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.GELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )

            self.pis = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden,self.hidden_hid),
            nn.GELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden_hid,self.gaussians)
            #nn.Softmax()
            ).to(device)

            # self.pis = nn.Linear(self.hidden_hid,self.gaussians).to(device)
            self.sigma_x = nn.Linear(self.hidden//4, self.gaussians).to(device)
            self.sigma_y = nn.Linear(self.hidden//4, self.gaussians).to(device)
            self.mu_x = nn.Linear(self.hidden//4, self.gaussians).to(device)
            self.mu_y = nn.Linear(self.hidden//4, self.gaussians).to(device)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        #mdn_embeded = self.embedding_mdn(decoder_output).to(device)
        # sigmax_embeded = self.embedding_sigma(decoder_output).to(device)
        # sigmay_embeded = self.embedding_sigma(decoder_output).to(device)
        # muex_embeded = self.embedding_mue(decoder_output).to(device)
        # muey_embeded = self.embedding_mue(decoder_output).to(device)

        # Calculate PI
        pi = self.pis(decoder_output).to(device)
        pi = nn.functional.softmax(pi, -1)

        # Calculate Sigmas
        sigma_embeded = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)
        mu_embeded = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)
        

        sigmax_embeded = self.embedding_sigma(sigma_embeded).to(device)
        sigmay_embeded = self.embedding_sigma(sigma_embeded).to(device)
        muex_embeded = self.embedding_mue(mu_embeded).to(device)
        muey_embeded = self.embedding_mue(mu_embeded).to(device)
        
        sigma_x = torch.Tensor(torch.exp(self.sigma_x(sigmax_embeded))).to(device)
        sigma_y = torch.Tensor(torch.exp(self.sigma_x(sigmay_embeded))).to(device)

        mu_x = torch.Tensor(self.mu_x(muex_embeded)).to(device)
        mu_y = torch.Tensor(self.mu_y(muey_embeded)).to(device)
       
        #result = torch.Tensor(decoder_output)

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output
    

class Attention_GMM_With_Device(nn.Module):
    def __init__(self,
        device,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        dropout=0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_GMM, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = dropout
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
            self.mdn_weight =torch.tensor(0.5)
            # self.mdn_weight = nn.Parameter(torch.tensor(0.5),requires_grad=True)
            #self.mdn_weight = nn.Parameter(torch.tensor([0.5]),requires_grad=True).to(device)
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )


            self.embedding_sigma = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )
            self.embedding_mue = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )

            self.pis = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU(),
            nn.Linear(self.hidden//4,self.gaussians)
            #nn.Softmax()
            )

            
            self.hidden_hid = self.hidden//4
            # self.pis = nn.Linear(self.hidden_hid,self.gaussians).to(device)
            self.sigma_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.sigma_y = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_y = nn.Linear(self.hidden_hid, self.gaussians)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        #mdn_embeded = self.embedding_mdn(decoder_output).to(device)
        sigmax_embeded = self.embedding_sigma(decoder_output).to(device)
        sigmay_embeded = self.embedding_sigma(decoder_output).to(device)
        muex_embeded = self.embedding_mue(decoder_output).to(device)
        muey_embeded = self.embedding_mue(decoder_output).to(device)

        # Calculate PI
        pi = self.pis(decoder_output).to(device)
        pi = nn.functional.softmax(pi, -1)

        # Calculate Sigmas
        sigma_x = torch.Tensor(torch.exp(self.sigma_x(sigmax_embeded))).to(device)
        sigma_y = torch.Tensor(torch.exp(self.sigma_x(sigmay_embeded))).to(device)

        mu_x = torch.Tensor(self.mu_x(muex_embeded)).to(device)
        mu_y = torch.Tensor(self.mu_y(muey_embeded)).to(device)
       
        #result = torch.Tensor(decoder_output)

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output
  
if __name__ == "__main__":
     # Train transformer only
    in_features = 2
    out_features = 3
    num_heads = CFG.num_heads
    num_encoder_layers = CFG.num_encoder_layers
    num_decoder_layers =  CFG.num_decoder_layers
    embedding_size = CFG.embd_size
    max_length = 8
    n_hidden = 10
    gaussians = 5
    forecast_window = 12
    drp=0.2
    # transformer_mdn = Transformer_MDN( 
    #         device,
    #         in_features,
    #         out_features,
    #         num_heads,
    #         num_encoder_layers,
    #         num_decoder_layers,
    #         embedding_size,
    #     ).to (device)
    attn_mdn = Attention_GMM_Encoder(device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size,n_gaussians=gaussians,n_hidden = n_hidden, dropout=drp).to(device)

    #attn_mdn = Attention_GMM(device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size).to(device)
    #print(transformer_mdn)
    for name, child in attn_mdn.named_children():
        print(name, child)
    inp = torch.rand(16,7,2).to(device)
    tgt = torch.rand(16,12,3).to(device)
    tgt_mask = generate_square_mask(dim_trg = 12 ,dim_src = 12, mask_type="tgt").to(device)


    pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = attn_mdn(inp,tgt,tgt_mask = tgt_mask)
    print('pi shape',pi.shape,'\nsigma_x shape:',sigma_x.shape,'\nmu_x shape: ',mu_x.shape)