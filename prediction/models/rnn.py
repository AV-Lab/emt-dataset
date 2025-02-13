#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:52 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluation.distance_metrics import calculate_ade, calculate_fde


class RNNPredictor:    
    class MultiEncoder(nn.Module):    
        def __init__(self, input_size, hidden_size, num_layers):
            """
            Initializes the LSTM encoder.
            
            Args:
                input_size (int): Number of input features (here, 2 for velocity).
                hidden_size (int): Number of hidden units in LSTM.
                num_layers (int): Number of LSTM layers.
            """
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        def forward(self, x):
            """
            Forward pass for the encoder.
            
            Args:
                x (Tensor): shape (batch_size, seq_len, input_size).
            
            Returns:
                hidden (Tensor): hidden states from the LSTM.
                cell   (Tensor): cell states from the LSTM.
            """
            outputs, (hidden, cell) = self.lstm(x)
            return hidden, cell
    
    class MultiDecoder(nn.Module):    
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            """
            Initializes the LSTM decoder.
            
            Args:
                input_size (int): Number of input features (2 for velocity).
                hidden_size (int): Number of hidden units in LSTM.
                output_size (int): Number of output features (2 for velocity).
                num_layers (int): Number of LSTM layers.
            """
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden, cell):
            """
            Forward pass for the decoder step-by-step (auto-regressive).
            
            Args:
                x (Tensor): shape (batch_size, 1, input_size).
                hidden (Tensor): hidden states from the LSTM encoder.
                cell   (Tensor): cell states from the LSTM encoder.
            
            Returns:
                predictions (Tensor): shape (batch_size, 1, output_size).
                hidden (Tensor): updated hidden states.
                cell   (Tensor): updated cell states.
            """
            outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
            predictions = self.fc(outputs)  # shape: (batch_size, 1, output_size)
            return predictions, hidden, cell
    
    class Seq2Seq(nn.Module):    
        def __init__(self, encoder, decoder):
            """
            Initializes the Seq2Seq model.
            
            Args:
                encoder (MultiEncoder): Encoder network.
                decoder (MultiDecoder): Decoder network.
            """
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        
        def forward(self, source, target_len):
            """
            Forward pass for the Seq2Seq model.
            
            Args:
                source (Tensor): shape (batch_size, enc_seq_len, input_size).
                target_len (int): Length of the future sequence to predict.
            
            Returns:
                outputs (Tensor): shape (batch_size, target_len, output_size).
            """
            batch_size = source.size(0)
            hidden, cell = self.encoder(source)
            decoder_input = torch.zeros(batch_size, 1, source.size(2)).to(source.device)
            
            outputs = []
            for _ in range(target_len):
                out_step, hidden, cell = self.decoder(decoder_input, hidden, cell)
                outputs.append(out_step)
                decoder_input = out_step
            
            # Concatenate the predictions along the time dimension
            outputs = torch.cat(outputs, dim=1)  # (batch_size, target_len, output_size)
            return outputs

    def __init__(self, observation_length, prediction_horizon, device, normalize, checkpoint_file=None):
        """
        Initializes the RNNPredictor with LSTM-based Seq2Seq, plus training configuration.
        
        Args:
            observation_length (int): # of observed timesteps (for reference).
            prediction_horizon (int): # of timesteps to predict into the future.
            checkpoint_file (str, optional): Path to a saved checkpoint.
        """
        # Model hyperparameters
        self.hidden_size = 128
        self.num_layers = 2
        
        # We use velocity dimension = 2 as input_size/output_size
        self.input_size = 2
        self.output_size = 2
        
        # Data sequence lengths (for reference)
        self.input_len = observation_length
        self.target_len = prediction_horizon
        
        # Training config
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.patience = 5
        self.device = device
        self.normalize = normalize
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Build model
        encoder = self.MultiEncoder(self.input_size, self.hidden_size, self.num_layers)
        decoder = self.MultiDecoder(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.model = self.Seq2Seq(encoder, decoder).to(self.device)
        
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Optionally load checkpoint
        if checkpoint_file is not None:
            print('Loading weights from checkpoint')
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'mean' in checkpoint:
                self.mean = checkpoint['mean']
                self.std  = checkpoint['std']

    def save_checkpoint(self, saving_checkpoint_path):
        """
        Saves the current model and optimizer states to a checkpoint file.
        
        Args:
            saving_checkpoint_path (str): Path to save the checkpoint (directory).
        """
        print("Saving the checkpoint ...")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.normalize:
            checkpoint['mean'] = self.mean
            checkpoint['std'] =  self.std
            
        torch.save(checkpoint, f'{saving_checkpoint_path}/trained_model.pth')
    
    def train(self, train_loader, valid_loader=None, saving_checkpoint_path=None):
        """
        Trains the model on normalized velocities. Also does optional validation.
        
        Args:
            train_loader (DataLoader): yields (obs_tensor, target_tensor) pairs.
                Expecting shape: [batch, obs_seq_len, 4], where
                    - columns [0:2] = absolute positions
                    - columns [2:4] = velocities
            valid_loader (DataLoader, optional): for validation
            saving_checkpoint_path (str, optional): where to save best checkpoint
            mean (Tensor): shape [4], mean of data columns.
            std  (Tensor): shape [4],  std of data columns.
        """
        print("Total train batches:", len(train_loader))
        
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device)
            self.std  = train_loader.dataset.std.to(self.device)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            
            epoch_loss = 0.0
            epoch_ade = 0.0
            epoch_fde = 0.0
            
            load_train = tqdm(train_loader, desc=f"Epoch: {epoch+1}/{self.num_epochs}", leave=False)
    
            for id_b, batch in enumerate(load_train):
                obs_tensor, target_tensor = batch
                obs_tensor = obs_tensor.to(self.device)  # shape [B, obs_seq_len, 4]
                target_tensor = target_tensor.to(self.device)  # shape [B, pred_seq_len, 4]
                
                if self.normalize:
                    # Input velocities and target velocities
                    input_vel = obs_tensor[:, 1:, 2:4]  # skip first frame’s velocity
                    input_vel_norm = (input_vel - self.mean[2:]) / self.std[2:]
                    target_vel = target_tensor[:, :, 2:4]
                    target_vel_norm = (target_vel - self.mean[2:]) / self.std[2:]
                else:
                    input_vel_norm = obs_tensor[:, 1:, 2:4]
                    target_vel_norm = target_tensor[:, :, 2:4]
                    
    
                self.optimizer.zero_grad()
                pred_vel_norm = self.model(input_vel_norm, target_vel_norm.shape[1])
                loss = self.criterion(pred_vel_norm, target_vel_norm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                
                # Compute ADE / FDE on Absolute Positions
                with torch.no_grad():
                    if self.normalize:
                        pred_vel = pred_vel_norm * self.std[2:] + self.mean[2:] #Unnormalize predicted velocities
                    else:
                        pred_vel = pred_vel_norm  # [B, pred_seq_len, 2]
                    
                    # Convert velocities -> absolute positions starting from the last observed absolute position
                    last_obs_pos = obs_tensor[:, -1, 0:2]   
                    B, T, _ = pred_vel.shape
                    pred_positions = torch.zeros(B, T, 2, device=self.device)
                    pred_positions[:, 0, :] = last_obs_pos + pred_vel[:, 0, :] # first predicted position:
                    
                    for t in range(1, T):
                        pred_positions[:, t, :] = pred_positions[:, t-1, :] + pred_vel[:, t, :]
                    
                    target_positions = target_tensor[:, :, 0:2]  # Ground-truth absolute positions
                    
                    ade_batch = calculate_ade(pred_positions, target_positions)
                    fde_batch = calculate_fde(pred_positions, target_positions)
                    epoch_ade += ade_batch
                    epoch_fde += fde_batch
                    
                # Update your progress bar
                load_train.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ADE':  f"{ade_batch:.4f}",
                    'FDE':  f"{fde_batch:.4f}"
                })
                
    
                if valid_loader is not None:
                    val_loss = self.validate(valid_loader)
                    
                    # Check for improvement
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        
                        # Optionally save the best model so far
                        if saving_checkpoint_path is not None:
                            self.save_checkpoint(saving_checkpoint_path)
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Early stopping
                    if self.epochs_without_improvement >= self.patience:
                        print("Early stopping triggered.")
                        break
                    
            # Average the losses/metrics over the entire epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_ade  = epoch_ade  / len(train_loader)
            avg_epoch_fde  = epoch_fde  / len(train_loader)
    
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train - Loss: {avg_epoch_loss:.4f}, ADE: {avg_epoch_ade:.4f}, FDE: {avg_epoch_fde:.4f}")    
        
        # If no validation loader we save the last model
        if saving_checkpoint_path is not None and valid_loader is None:
            self.save_checkpoint(saving_checkpoint_path)

    def validate(self, valid_loader):
        """
        Validates the model. If self.normalize == True, we compute MSE loss on
        normalized velocities. Otherwise, we compute MSE on raw velocities.
        
        Returns:
            float: Average validation MSE loss on velocities (normalized or raw).
        """
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for obs_tensor, target_tensor in valid_loader:
                obs_tensor    = obs_tensor.to(self.device)  
                target_tensor = target_tensor.to(self.device) 

                if self.normalize:
                    input_vel_norm = (obs_tensor[:, 1:, 2:4] - self.mean[2:]) / self.std[2:]
                    target_vel_norm = (target_tensor[:, :, 2:4] - self.mean[2:]) / self.std[2:]
                    pred_vel_norm = self.model(input_vel_norm, target_vel_norm.shape[1])
                    loss = self.criterion(pred_vel_norm, target_vel_norm)
                else:
                    input_vel = obs_tensor[:, 1:, 2:4]    
                    target_vel = target_tensor[:, :, 2:4]
                    pred_vel = self.model(input_vel, target_vel.shape[1])
                    loss = self.criterion(pred_vel, target_vel)
  
                epoch_loss += loss.item()
        
        avg_val_loss = epoch_loss / len(valid_loader)
        print(f'Validation Loss (velocity-space): {avg_val_loss:.4f}')
        return avg_val_loss
    
    
    def evaluate(self, loader):
        """
        Evaluates the model by computing ADE/FDE in absolute position space.
        
        If self.normalize == True, we:
          1) Normalize input velocities,
          2) Get predicted normalized velocities,
          3) Unnormalize them to get raw velocities,
          4) Cumulatively sum to get absolute positions.
        
        If self.normalize == False, we skip all normalization steps.
        
        Returns:
            float: ADE
            float: FDE
        """
        self.model.eval()
        ade = 0.0
        fde = 0.0
        num_batches = len(loader)
        
        print("Total test batches:", num_batches)
        
        with torch.no_grad():
            for obs_tensor, target_tensor in loader:
                obs_tensor    = obs_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                if self.normalize:
                    input_vel_norm = (obs_tensor[:, 1:, 2:4] - self.mean[2:]) / self.std[2:]
                    pred_vel_norm = self.model(input_vel_norm, target_tensor.shape[1])
                    pred_vel = pred_vel_norm * self.std[2:] + self.mean[2:]  
                else:
                    input_vel = obs_tensor[:, 1:, 2:4]
                    pred_vel  = self.model(input_vel, target_tensor.shape[1])    
                
                # Reconstruct predicted absolute positions
                last_obs_pos = obs_tensor[:, -1, 0:2]  # [B, 2]
                B, T, _ = pred_vel.shape
                pred_positions = torch.zeros((B, T, 2), device=self.device)
                pred_positions[:, 0, :] = last_obs_pos + pred_vel[:, 0, :] # First future position

                for t in range(1, T):
                    pred_positions[:, t, :] = pred_positions[:, t-1, :] + pred_vel[:, t, :]
                
                target_positions = target_tensor[:, :, 0:2]
                
                ade += calculate_ade(pred_positions, target_positions)
                fde += calculate_fde(pred_positions, target_positions)
        
        ade = ade / num_batches
        fde = fde / num_batches
        print(f"Evaluation --> ADE: {ade:.4f}, FDE: {fde:.4f}")
        return ade, fde
    
    
    def predict(self, obs_tensor, target_len):
        """
        Predict future absolute positions given obs_tensor.
        
        If self.normalize == True, we:
          1) Normalize the input velocities,
          2) Predict normalized velocities,
          3) Unnormalize them,
          4) Cumulatively sum to get positions.
        
        Otherwise, we skip normalization.
        
        Returns:
            Tensor of shape [B, target_len, 2] for predicted absolute positions.
        """
        self.model.eval()
        
        obs_tensor = obs_tensor.to(self.device)
        with torch.no_grad():
            if self.normalize:
                input_vel_norm = (obs_tensor[:, 1:, 2:4] - self.mean[2:]) / self.std[2:]
                pred_vel_norm = self.model(input_vel_norm, target_len)
                pred_vel = pred_vel_norm * self.std[2:] + self.mean[2:]
            else:
                input_vel = obs_tensor[:, 1:, 2:4]
                pred_vel  = self.model(input_vel, target_len)

            last_obs_pos = obs_tensor[:, -1, 0:2]  # [B, 2]
            B, T, _ = pred_vel.shape
            
            pred_positions = torch.zeros((B, T, 2), device=self.device)
            pred_positions[:, 0, :] = last_obs_pos + pred_vel[:, 0, :]
            
            for t in range(1, T):
                pred_positions[:, t, :] = pred_positions[:, t-1, :] + pred_vel[:, t, :]
        
        return pred_positions