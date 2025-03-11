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
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
            predictions = self.fc(outputs)  
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
            
            outputs = torch.cat(outputs, dim=1) 
            return outputs

    def __init__(self, observation_length, prediction_horizon, device, normalize=False, checkpoint_file=None):
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
                self.normalize = True
                print("please note model was trained on normalized values => self.normalize is set to True")

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
            
        torch.save(checkpoint, f'{saving_checkpoint_path}/rnn_trained_model.pth')
    
    def train(self, train_loader, valid_loader=None, save_path=None):
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
                obs_tensor = obs_tensor.to(self.device)  
                target_tensor = target_tensor.to(self.device)  
                
                if self.normalize:
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
                        pred_vel = pred_vel_norm * self.std[2:] + self.mean[2:] 
                    else:
                        pred_vel = pred_vel_norm  
                    
                    # Convert velocities -> absolute positions starting from the last observed absolute position
                    last_obs_pos = obs_tensor[:, -1, 0:2]   
                    B, T, _ = pred_vel.shape
                    pred_positions = torch.zeros(B, T, 2, device=self.device)
                    pred_positions[:, 0, :] = last_obs_pos + pred_vel[:, 0, :] 
                    
                    for t in range(1, T):
                        pred_positions[:, t, :] = pred_positions[:, t-1, :] + pred_vel[:, t, :]
                    
                    target_positions = target_tensor[:, :, 0:2]  
                    
                    ade_batch = calculate_ade(pred_positions, target_positions)
                    fde_batch = calculate_fde(pred_positions, target_positions)
                    epoch_ade += ade_batch
                    epoch_fde += fde_batch
                    
                load_train.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ADE':  f"{ade_batch:.4f}",
                    'FDE':  f"{fde_batch:.4f}"
                })
                    
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_ade  = epoch_ade  / len(train_loader)
            avg_epoch_fde  = epoch_fde  / len(train_loader)
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train - Loss: {avg_epoch_loss:.4f}, ADE: {avg_epoch_ade:.4f}, FDE: {avg_epoch_fde:.4f}")   
            
            
            if valid_loader is not None:
                val_loss = self.validate(valid_loader)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    if save_path is not None:
                        self.save_checkpoint(save_path)
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        # If no validation loader we save the last model
        if save_path is not None and valid_loader is None:
            self.save_checkpoint(save_path)

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
                
                last_obs_pos = obs_tensor[:, -1, 0:2]
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
    
    
    def predict(self, obs_tensors):
        """
        Predict future absolute positions given a list of observed trajectories.
        
        Each observation in obs_tensors is a list or array of absolute positions [x, y]
        per agent. The function:
          1) Computes velocities as differences between consecutive positions.
          2) Pads the velocity sequence at the beginning if it's shorter than (observation_len - 1).
          3) Stacks the velocities into a batch and predicts future velocities using self.model.
          4) Reconstructs the absolute positions by cumulatively summing the predicted velocities,
             starting from the last observed absolute position.
        
        Returns:
            A list of predictions, where each element is a NumPy array of shape [target_len, 2]
            representing the predicted future absolute positions for that agent.
        """
        self.model.eval()
        processed_velocities = []
    
        for trajectory in obs_tensors:
            traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=self.device)
            
            if traj_tensor.shape[0] == 1:
                processed_velocities.append(
                    torch.zeros((self.input_len - 1, 2), dtype=torch.float32, device=self.device)
                )
                continue
            
            obs_vel = traj_tensor[1:] - traj_tensor[:-1]
            
            if obs_vel.shape[0] < self.input_len - 1:
                pad_size = self.input_len - 1 - obs_vel.shape[0]
                pad = torch.zeros((pad_size, 2), dtype=torch.float32, device=self.device)
                obs_vel = torch.cat([pad, obs_vel], dim=0)
            
            processed_velocities.append(obs_vel)
        
        velocities_batch = torch.stack(processed_velocities, dim=0)
        
        with torch.no_grad():
            if self.normalize:
                input_vel_norm = (velocities_batch - self.mean[2:]) / self.std[2:]
                pred_vel_norm = self.model(input_vel_norm, self.target_len)
                pred_vel = pred_vel_norm * self.std[2:] + self.mean[2:]
            else:
                pred_vel = self.model(velocities_batch, self.target_len)
        
        # Reconstruct absolute positions from the predicted velocities.
        pred_positions_batch = []
        for i, trajectory in enumerate(obs_tensors):
            traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=self.device)
            last_obs_pos = traj_tensor[-1, 0:2]  
            pred_vel_i = pred_vel[i]          
            
            # Initialize predicted positions.
            pred_positions = torch.zeros((self.target_len, 2), dtype=torch.float32, device=self.device)
            pred_positions[0] = last_obs_pos + pred_vel_i[0]
            for t in range(1, self.target_len):
                pred_positions[t] = pred_positions[t - 1] + pred_vel_i[t]        
            pred_positions_batch.append(pred_positions)
    
        predictions = [pred.cpu().numpy() for pred in pred_positions_batch]
        return predictions
