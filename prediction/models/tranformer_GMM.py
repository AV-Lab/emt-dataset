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

from typing import Tuple, Dict, Optional, List, Union,Any
from dataclasses import dataclass, asdict
from evaluation.distance_metrics import calculate_ade,calculate_fde
from utils import set_seeds
from tqdm import tqdm
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
        """Update running metrics with batch results and store history"""
        for key, value in metrics_dict.items():
            self.running_metrics[phase][key]['value'] += value * batch_size
            self.running_metrics[phase][key]['count'] += batch_size

        # If all batches are processed (end of epoch), store history
        if phase == 'train':
            epoch_metrics = self.get_averages(phase='train')
            self.history['train_loss'].append(epoch_metrics['loss'])
            self.history['train_ade'].append(epoch_metrics['ade'])
            self.history['train_fde'].append(epoch_metrics['fde'])
            self.history['train_best_ade'].append(epoch_metrics['best_ade'])
            self.history['train_best_fde'].append(epoch_metrics['best_fde'])
        
        elif phase == 'test':
            epoch_metrics = self.get_averages(phase='test')
            self.history['test_loss'].append(epoch_metrics['loss'])
            self.history['test_ade'].append(epoch_metrics['ade'])
            self.history['test_fde'].append(epoch_metrics['fde'])
            self.history['test_best_ade'].append(epoch_metrics['best_ade'])
            self.history['test_best_fde'].append(epoch_metrics['best_fde'])


    def get_averages(self, phase='train'):
        """Compute averages for specified phase."""
        if phase not in self.running_metrics:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train' or 'test'.")

        return {
            key: (metric['value'] / metric['count'] if metric['count'] > 0 else 0)
            for key, metric in self.running_metrics[phase].items()
        }

    def reset(self, phase='train'):
        """Reset running metrics for specified phase."""
        self.running_metrics[phase] = self._init_metric_dict()

    def record_history(self, phase='train'):
        """Store epoch metrics for future analysis."""
        avg_metrics = self.get_averages(phase)
        for key in avg_metrics:
            self.history[f"{phase}_{key}"].append(avg_metrics[key])

    def print_epoch_metrics(self, epoch, epochs, verbose=True):
        """Print epoch metrics including best-of-N results."""
        if not verbose:
            return

        train_avgs = self.get_averages('train')
        test_avgs = self.get_averages('test')

        # print(f"\nEpoch {epoch+1}/{epochs}")

        if self.train_available:
            print(f"Train - Loss: {train_avgs['loss']:.4f}, "
                  f"ADE: {train_avgs['ade']:.4f}, "
                  f"FDE: {train_avgs['fde']:.4f}, "
                  f"Best_ADE: {train_avgs['best_ade']:.4f}, "
                  f"Best_FDE: {train_avgs['best_fde']:.4f}")

        if self.test_available:
            print(f"Test  - Loss: {test_avgs['loss']:.4f}, "
                  f"ADE: {test_avgs['ade']:.4f}, "
                  f"FDE: {test_avgs['fde']:.4f}, "
                  f"Best_ADE: {test_avgs['best_ade']:.4f}, "
                  f"Best_FDE: {test_avgs['best_fde']:.4f}")


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
    

    # GMM parameters
    n_gaussians: int = 6
    n_hidden: int = 32

    # Optimizer parameters
    lr_mul: float = 0.2
    n_warmup_steps: int = 4000
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-9

    def __post_init__(self):
        """Post-init processing."""
        if self.checkpoint_file is None:
            self.checkpoint_file = f'GMM_transformer_P_{self.past_trajectory}_F_{self.future_trajectory}_W_x.pth'

    def get_device(self) -> torch.device:
        """Return the device for computation."""
        return self.device if self.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def display_config(self, verbose: bool = False) -> None:
        """
        Pretty print the model configuration.
        
        Args:
            verbose (bool): If True, prints additional information and formatting
        """
        if verbose:
            print("\n" + "="*50)
            print("AttentionGMM Model Configuration")
            print("="*50)
            
            print("\nModel Architecture:")
            print("-"*20)
            print(f"Input Features:      {self.in_features}")
            print(f"Output Features:     {self.out_features}")
            print(f"Number of Heads:     {self.num_heads}")
            print(f"Encoder Layers:      {self.num_encoder_layers}")
            print(f"Decoder Layers:      {self.num_decoder_layers}")
            print(f"Embedding Size:      {self.embedding_size}")
            print(f"Dropout Rate:        {self.dropout}")
            print(f"Batch First:         {self.batch_first}")
            print(f"Activation Function: {self.actn}")
            print("\nGMM Settings:")
            print("-"*20)
            print(f"Number of Gaussians: {self.n_gaussians}")
            print(f"Hidden Size:         {self.n_hidden}")
            
            print("\nOptimizer Settings:")
            print("-"*20)
            print(f"Learning Rate Multiplier: {self.lr_mul}")
            print(f"Warmup Steps:            {self.n_warmup_steps}")
            print(f"Optimizer Betas:         {self.optimizer_betas}")
            print(f"Optimizer Epsilon:       {self.optimizer_eps}")
            
            print("\nDevice Configuration:")
            print("-"*20)
            print(f"Device: {self.get_device()}")
            print("\n" + "="*50)
        else:
            # Simple print of key parameters
            print(f"AttentionGMM Config: in_features={self.in_features}, "
                  f"out_features={self.out_features}, num_heads={self.num_heads}, "
                  f"embedding_size={self.embedding_size}, dropout={self.dropout}, "
                  f"n_gaussians={self.n_gaussians}")

    def get_device(self) -> torch.device:
        """Get the device to use. If not provided, use cuda if available else cpu."""
        if self.device is not None:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)   
     
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

        # Use provided config or create from kwargs
        # self.config = config or ModelConfig(**kwargs)

        # Create config object first
        if config is None:
            self.config = ModelConfig(**kwargs)  # Create from kwargs, using defaults for unspecified params
        else:
            self.config = config
        
        # Initialize metric tracker here
        self.tracker = MetricTracker()
        # Checkpoint file (user-defined or default)
        self.checkpoint_file = self.config.checkpoint_file


        
        # Store model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #self.config.get_device # Uses either provided device or default
        self.num_heads = self.config.num_heads
        self.device = self.config.get_device()
        self.max_len = max(self.config.past_trajectory, self.config.future_trajectory)
        self.mean = self.config.mean.to(self.device)
        self.std = self.config.std.to(self.device)
        self.normalized = self.config.normalize
        self.d_model = self.config.embedding_size
        self.input_features = self.config.in_features
        self.output_features = self.config.out_features

        # Store optimizer parameters
        self.lr_mul = self.config.lr_mul 
        self.n_warmup_steps = self.config.n_warmup_steps
        self.optimizer_betas = self.config.optimizer_betas
        self.optimizer_eps = self.config.optimizer_eps


        self.gaussians =  self.config.n_gaussians
        self.hidden = self.config.n_hidden
        
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


        # Create independent embedding networks  for mu and sigma
        self.embedding_sigma = self.create_gmm_embedding()
        self.embedding_mue = self.create_gmm_embedding()
        self.embedding_pi = self.create_gmm_embedding()

        # Mixture weights
        self.pis_head = nn.Linear(self.hidden // 2, self.gaussians)
        # Output layers for sigma and mu
        self.sigma_head = nn.Linear(self.hidden // 2, self.gaussians * 2)
        self.mu_head = nn.Linear(self.hidden // 2, self.gaussians * 2)

        
    def create_gmm_embedding(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, int(self.hidden * 0.75)),
            nn.ELU(),
            nn.Linear(int(self.hidden * 0.75), self.hidden // 2),
            nn.ELU()
        )
    def _init_weights(self):
        """Initialize the model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None) -> 'AttentionGMM':
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
        
        for b in range(batch_size):
            for t in range(seq_len):
                errors = []
                for k in range(n_mixtures):
                    pred = mue[b, t, k]
                    error = torch.norm(pred - gt_normalized[b, t])
                    errors.append(error)
                
                best_k = torch.argmin(torch.tensor(errors))
                best_of_n_pred[b, t] = mue[b, t, best_k]
        
        return highest_prob_pred, best_of_n_pred
    
    
    def _bivariate(self,pi,sigma_x,sigma_y, mu_x , mu_y,input):

        # Check the num of dims
        if input.ndim ==3:
            x = input[:,:,0]
            y = input[:,:,1]
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            #print("Num of Dims is 3 : ",input.shape)
        elif input.ndim ==2:
            x = input[:,0]
            y = input[:,1]
            x = x.unsqueeze(dim=1)
            y = y.unsqueeze(dim=1)
            # print("Num of Dims is 2 : ",input.shape)
        # make |mu|=K copies of y, subtract mu, divide by sigma
        #print("Input: ",input.shape ,"\nX: ",x.shape,"\nY: ",y.shape,"\nMu_x : ",mu_x.shape,"\nMu_y : ",mu_y.shape,"\nSigma_x : ",sigma_x.shape)
        result_x = torch.square((x.expand_as(mu_x) - mu_x) * torch.reciprocal(sigma_x))
        result_y = torch.square((y.expand_as(mu_y) - mu_y) * torch.reciprocal(sigma_y))
        

        result = -0.5*(result_x + result_y)
        log_pi = torch.log(pi)
        log_TwoPiSigma = -torch.log (2.0*np.pi*sigma_x*sigma_y)
        # expand log values
        values = log_pi + log_TwoPiSigma.expand_as(log_pi) 

        return (values + result)
    def _mdn_loss_fn(self,pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures):
        # calculate the score for each mixture of the gaussian_distribution
        # input shape (sample_size,num_mixtures,parameter) parametr is 2 in mue (x,y) and 2,2 in sigma [xx,xy,yx,yy] 
        # Pi has shape of  (sample_size,num_mixtures)
        # swap axis to have shape (num_mixtures,sample_size,parameter)
        # print("Before anythinG: ",sigma_x.shape,sigma_y.shape, mu_x.shape , mu_y.shape,'\n: ',y.shape,'\n')


        # mask = torch.lt(pi, 0)
        # mask_res = torch.lt(pi, 0)

        # # check if any element in the tensor satisfies the condition
        # if torch.any(mask):
        #     print("The pi tensor contains negative values.")
        # else:
        #     print("The pi tensor does not contain negative values.")

        
        result = self._bivariate(pi,sigma_x,sigma_y, mu_x , mu_y,y) 
        # print("result shape: ",result.shape)
        # mask_res = torch.lt(result, 0)

        # # check if any element in the tensor satisfies the condition
        # if torch.any(mask_res):
        #     print("The result tensor contains negative values.")
        # else:
        #     print("The result tensor does not contain negative values.")
        # max of results
        # m = torch.max(result)
        # changed value of max
        #torch.tensor
        m = (torch.max(result, dim=2, keepdim=True)[0]).repeat(1,1,mixtures)
        # print("max of results shape: ",m.shape)
        # print("result of results shape: ",result.shape)
        # LogSumExp trick log(sum(exp)) will be = m + log(sum (exp (result-m)))
        exp_value = torch.exp(result-m)
        # print("exp_value of exp_value shape: ",exp_value.shape)
        epsilon = 0.00001
        # changed the last dimention dim from 1 to -1
        result = torch.sum(exp_value, dim=-1) + epsilon
        #print("result after sum: ",result)
        #org
        #result = -(m + torch.log(result))
        result = -(m[:,:,0] + torch.log(result))
        # counter+=1
        if(torch.isnan(result).any()):
            # print("Counter loss: ",counter)
            print("result m : ",m.item)
        return torch.mean(result)
    def train(
        self,
        train_dl: DataLoader,
        test_dl: DataLoader = None,
        epochs: int = 100,
        verbose: bool = True,
        save_path: str = 'prediction/results',
        save_model: bool = True,
        save_frequency: int = 100,
        # checkpoint_name: str = 'model.pt'
    ) -> Tuple[nn.Module, Dict]:
        """
        Train the model with metrics tracking and visualization.
        """
        # Initialize weights using Xavier uniform initialization
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
                train_loss = self._mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,target,self.config.n_gaussians)
                
                # Backward pass
                train_loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20.0)
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

            # Test evaluation
            if test_dl is not None:
                self.evaluate(test_dl,from_train=True)

            # Print epoch metrics
            self.tracker.print_epoch_metrics(epoch, epochs, verbose)

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
                # torch.save(model_state, os.path.join(models_dir, checkpoint_name))
                # Save the model
                checkpoint_name = self.checkpoint
                os.makedirs(save_path, exist_ok=True)
                torch.save(model_state, os.path.join(models_dir, f"{checkpoint_name}"))
                print("saving checkpoint at : ", os.path.join(models_dir, f"{checkpoint_name}"))


            # Reset metrics for next epoch
            self.tracker.reset('train')
            self.tracker.reset('test')
        # Plot training history if verbose
        if verbose:
            self.plot_metrics(
                self.tracker.history['train_loss'], self.tracker.history['test_loss'],
                self.tracker.history['train_ade'], self.tracker.history['test_ade'],
                self.tracker.history['train_fde'], self.tracker.history['test_fde'],
                enc_seq_len, dec_seq_len, batch_size,
                save_path=metrics_dir
            )

        return self, self.tracker.history


    # @staticmethod
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

    def evaluate(self,test_loader=None,from_train=False):
        """
        Evaluate the model on test data
        Args:
            test_loader: DataLoader for test data
            from_train: Boolean indicating if called during training
        """
        
        # Need to use nn.Module's train method for mode setting
        super().train(False)  # Same as eval() but avoids the conflict
        self.tracker.test_available = True
        with torch.no_grad():
            for batch in test_loader:
                obs_tensor_eval, target_tensor_eval = batch
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
                eval_loss = self._mdn_loss_fn(pi_eval, sigma_x_eval,sigma_y_eval, mu_x_eval , mu_y_eval,target_eval,self.config.n_gaussians)
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
                self.tracker.update(batch_metrics, obs_tensor_eval.shape[0], phase='test')

        # Print epoch metrics
        if not from_train:
            self.tracker.print_epoch_metrics(epoch=1, epochs=1, verbose=True)
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



