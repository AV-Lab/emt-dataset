#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:08:44 2025

@author: nadya
"""


import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from evaluation.distance_metrics import calculate_ade, calculate_fde
from dataloaders.frame_loader import compute_adjacency_matrix



def masked_mse_loss(pred, target, mask):
    """
    Compute the Mean Squared Error (MSE) loss on valid nodes only.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, T, N, out_dim).
        target (torch.Tensor): Ground-truth tensor of shape (B, T, N, out_dim).
        mask (torch.Tensor): Binary mask tensor of shape (B, N) with 1.0 for valid nodes and 0.0 for padded nodes.
    
    Returns:
        torch.Tensor: Scalar loss value averaged over valid elements.
    """
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
    loss = (pred - target) ** 2
    loss = loss * mask_expanded
    return loss.sum() / mask_expanded.sum()

class GCNLayer(nn.Module):
    """
    A Graph Convolutional Network (GCN) layer.
    
    This layer performs a linear transformation followed by aggregation using the adjacency matrix.
    
    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
    
    Attributes:
        weight (nn.Parameter): Learnable weight matrix of shape (in_features, out_features).
        bias (nn.Parameter, optional): Learnable bias vector of shape (out_features).
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize learnable parameters using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass of the GCN layer.
        
        Args:
            x (torch.Tensor): Input features tensor of shape (B, N, in_features).
            adj (torch.Tensor): Adjacency matrix tensor of shape (B, N, N).
        
        Returns:
            torch.Tensor: Output features tensor of shape (B, N, out_features).
        """
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out += self.bias
        return out

class GCNEncoder(nn.Module):
    """
    Encoder module based on stacked GCN layers.
    
    Args:
        in_features (int): Number of input features per node.
        hidden_size (int): Number of output features per node after encoding.
        num_layers (int, optional): Number of GCN layers to stack. Default is 2.
    
    Returns:
        torch.Tensor: Encoded node features of shape (B, N, hidden_size).
    """
    def __init__(self, in_features, hidden_size, num_layers=2):
        super().__init__()
        layers = []
        layers.append(GCNLayer(in_features, hidden_size))
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_size, hidden_size))
        self.gcn_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
    
    def forward(self, x, adj):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input features of shape (B, N, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (B, N, N).
        
        Returns:
            torch.Tensor: Encoded features of shape (B, N, hidden_size).
        """
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.activation(x)
        return x

class GCNDecoder(nn.Module):
    """
    Decoder module based on stacked GCN layers.
    
    Args:
        hidden_size (int): Number of input features per node (from encoder).
        out_size (int): Desired number of output features per node.
        num_layers (int, optional): Number of GCN layers to stack. Default is 2.
    
    Returns:
        torch.Tensor: Decoded features of shape (B, N, out_size).
    """
    def __init__(self, hidden_size, out_size, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_size, hidden_size))
        layers.append(GCNLayer(hidden_size, out_size))
        self.gcn_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
    
    def forward(self, x, adj):
        """
        Forward pass of the decoder.
        
        Args:
            x (torch.Tensor): Encoded features of shape (B, N, hidden_size).
            adj (torch.Tensor): Adjacency matrix of shape (B, N, N).
        
        Returns:
            torch.Tensor: Decoded output of shape (B, N, out_size).
        """
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, adj)
            if i < len(self.gcn_layers) - 1:
                x = self.activation(x)
        return x

class GCNSeq2Seq(nn.Module):
    """
    A simple GCN-based Seq2Seq model.
    
    This model uses an encoder to process the observed features and a decoder to generate a sequence of predictions.
    
    Args:
        encoder (nn.Module): The encoder module (e.g., GCNEncoder).
        decoder (nn.Module): The decoder module (e.g., GCNDecoder).
    
    Returns:
        torch.Tensor: Sequence of predictions with shape (B, future_len, N, out_size).
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, feats, adj, future_len):
        """
        Forward pass of the Seq2Seq model.
        
        Args:
            feats (torch.Tensor): Input features of shape (B, N, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (B, N, N).
            future_len (int): Number of future time steps to predict.
        
        Returns:
            torch.Tensor: Predicted sequence of shape (B, future_len, N, out_size).
        """
        latent = self.encoder(feats, adj)
        outs = []
        for _ in range(future_len):
            decoded = self.decoder(latent, adj)
            outs.append(decoded.unsqueeze(1))
        return torch.cat(outs, dim=1)

class GCNPredictor:
    """
    GCN-based Seq2Seq predictor (without LSTM) that trains on velocity data [x, y, vx, vy].
    
    Data assumptions:
        - feats:   (B, T_obs, N, 4) where each feature is [x, y, vx, vy]
        - adj:     (B, N, N)
        - targets: (B, T_future, N, 4) where each target is [x, y, vx, vy] for future steps
        - mask:    (B, N)
    
    The model uses only the velocity channels (vx, vy) for prediction. For evaluation (ADE/FDE),
    the predicted velocities are integrated into absolute positions using the last observed positions.
    
    Args:
        observation_length (int): Number of observed time steps.
        future_length (int): Number of future time steps to predict.
        max_nodes (int): Maximum number of agents (nodes).
        device (str or torch.device): Computation device.
        normalize (bool): If True, applies normalization to (vx, vy) channels.
        checkpoint_file (str, optional): Path to a checkpoint file to load the model.
    """
    def __init__(self, observation_length, future_length, device, max_nodes=50, normalize=False, checkpoint_file=None):
        self.future_length = future_length
        self.input_len = observation_length  # T_obs
        self.device = device
        self.normalize = normalize
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.patience = 5
        self.max_nodes = max_nodes
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.hidden_size = 256
        self.num_layers = 2

        self.in_size = (observation_length - 1)* 2   
        self.out_size = 2

        encoder = GCNEncoder(self.in_size, self.hidden_size, self.num_layers)
        decoder = GCNDecoder(self.hidden_size, self.out_size, self.num_layers)
        self.model = GCNSeq2Seq(encoder, decoder).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = masked_mse_loss

        if checkpoint_file is not None:
            print(f"Loading checkpoint from {checkpoint_file}")
            ckpt = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.max_nodes = ckpt['max_nodes']
            if 'mean' in ckpt:
                self.mean = ckpt['mean']
                self.std  = ckpt['std']
                self.normalize = True
                print("Model was trained on normalized values. Setting self.normalize to True.")

    
    def save_checkpoint(self, folder):
        """
        Save the current model and optimizer state to a checkpoint file.
        
        Args:
            folder (str): Directory where the checkpoint will be saved.
        
        Returns:
            None
        """
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            ckpt["mean"] = self.mean
            ckpt["std"] = self.std
        ckpt["max_nodes"] = self.max_nodes
        torch.save(ckpt, f"{folder}/gcn_trained_model.pth")
        print(f"Checkpoint saved to {folder}/gcn_trained_model.pth")
    
    
    def train(self, train_loader, valid_loader=None, save_path=None):
        """
        Expects each batch to yield:
          feats:   (B, T_obs, N, 4)  -> [x, y, vx, vy]
          adj:     (B, N, N)
          targets: (B, T_future, N, 4) -> we use only targets[...,2:4]
          mask:    (B, N)
        """
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device).view(1, 1, 1, 4)
            self.std  = train_loader.dataset.std.to(self.device).view(1, 1, 1, 4)
    
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_ade  = 0.0
            epoch_fde  = 0.0
    
            load_train = tqdm(train_loader, desc=f"Epoch: {epoch+1}/{self.num_epochs}", leave=False)
            for id_b, batch in enumerate(load_train):
                feats, adj, targets, mask = batch
                feats   = feats.to(self.device)   
                adj     = adj.to(self.device)     
                targets = targets.to(self.device)    
                mask    = mask.to(self.device)        
                
                feats_vel   = feats[..., 2:4] 
                targets_vel = targets[..., 2:4]  
                self.optimizer.zero_grad()
                
                if self.normalize and self.mean is not None and self.std is not None:
                    feats_vel = (feats_vel - self.mean[...,2:4]) / self.std[...,2:4]
                    targets_vel = (targets_vel - self.mean[...,2:4]) / self.std[...,2:4]
                
                B, T_obs, N, C = feats_vel.shape
                flattened_dim = T_obs * C 
                feats_vel_flat = feats_vel.permute(0, 2, 1, 3).reshape(B, N, flattened_dim)
                pred_vel = self.model(feats_vel_flat, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
                
                with torch.no_grad():
                    if self.normalize:
                        pred_vel_un = pred_vel * self.std[...,2:4] + self.mean[...,2:4]
                    else:
                        pred_vel_un = pred_vel
                    
                    last_obs_pos = feats[:, -1, :, 0:2]  
                    B, T_pred, N, _ = pred_vel_un.shape
                    pred_positions = torch.zeros(B, T_pred, N, 2, device=self.device)
                    pred_positions[:, 0, :, :] = last_obs_pos + pred_vel_un[:, 0, :, :]
                    for t in range(1, T_pred):
                        pred_positions[:, t, :, :] = pred_positions[:, t-1, :, :] + pred_vel_un[:, t, :, :]
                    target_positions = targets[..., 0:2]
                    
                    predictions_list = []
                    targets_list = []
                    for b in range(B):
                        valid_idx = (mask[b] == 1).nonzero(as_tuple=False).squeeze()
                        if valid_idx.numel() == 0:
                            continue
                        predictions_list.append(pred_positions[b, :, valid_idx, :])
                        targets_list.append(target_positions[b, :, valid_idx, :])
                    if len(predictions_list) > 0:
                        ade_batch = calculate_ade(predictions_list, targets_list)
                        fde_batch = calculate_fde(predictions_list, targets_list)
                    else:
                        ade_batch, fde_batch = 0.0, 0.0
                    epoch_ade += ade_batch if isinstance(ade_batch, float) else ade_batch.item()
                    epoch_fde += fde_batch if isinstance(fde_batch, float) else fde_batch.item()
                
                load_train.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ADE': f"{ade_batch:.4f}",
                    'FDE': f"{fde_batch:.4f}"
                })
                
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_ade  = epoch_ade / len(train_loader)
            avg_epoch_fde  = epoch_fde / len(train_loader)
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
                
        if save_path is not None and valid_loader is None:
            self.save_checkpoint(save_path)

    def validate(self, valid_loader):
        """
        Validate the model on a validation dataset.
        
        Args:
            valid_loader (DataLoader): Yields (feats, adj, targets, mask).
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for feats, adj, targets, mask in valid_loader:
                feats   = feats.to(self.device)   
                adj     = adj.to(self.device)     
                targets = targets.to(self.device)  
                mask    = mask.to(self.device)        
                
                # Use velocity channels (vx, vy)
                feats_vel   = feats[..., 2:4]    
                targets_vel = targets[..., 2:4] 
                
                if self.normalize and self.mean is not None and self.std is not None:
                    feats_vel = (feats_vel - self.mean[..., 2:4]) / self.std[..., 2:4]
                    targets_vel = (targets_vel - self.mean[..., 2:4]) / self.std[..., 2:4]
                
                B, T_obs, N, C = feats_vel.shape
                flattened_dim = T_obs * C
                feats_vel_flat = feats_vel.permute(0, 2, 1, 3).reshape(B, N, flattened_dim)
                
                # Forward pass.
                pred_vel = self.model(feats_vel_flat, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()
        avg_val_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate(self, test_loader):
        """
        Evaluate the model on a test dataset and compute velocity MSE loss, ADE, and FDE.
        
        Args:
            test_loader (DataLoader): Yields (feats, adj, targets, mask).
        
        Returns:
            tuple: (average test loss, average ADE, average FDE)
        """
        self.model.eval()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for feats, adj, targets, mask in test_loader:
                feats   = feats.to(self.device)      
                adj     = adj.to(self.device)       
                targets = targets.to(self.device)      
                mask    = mask.to(self.device)        
                
                feats_vel   = feats[..., 2:4]    
                targets_vel = targets[..., 2:4] 
                
                if self.normalize and self.mean is not None and self.std is not None:
                    feats_vel = (feats_vel - self.mean[..., 2:4]) / self.std[..., 2:4]
                    targets_vel = (targets_vel - self.mean[..., 2:4]) / self.std[..., 2:4]
                
                B, T_obs, N, C = feats_vel.shape
                flattened_dim = T_obs * C
                feats_vel_flat = feats_vel.permute(0, 2, 1, 3).reshape(B, N, flattened_dim)
                
                pred_vel = self.model(feats_vel_flat, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()
                
                if self.normalize and self.mean is not None and self.std is not None:
                    pred_vel = pred_vel * self.std[..., 2:4] + self.mean[..., 2:4]
                
                # Reconstruct positions by integrating predicted velocities.
                last_obs_pos = feats[:, -1, :, 0:2] 
                B, T_pred, N, _ = pred_vel.shape
                pred_positions = torch.zeros(B, T_pred, N, 2, device=self.device)
                pred_positions[:, 0, :, :] = last_obs_pos + pred_vel[:, 0, :, :]
                for t in range(1, T_pred):
                    pred_positions[:, t, :, :] = pred_positions[:, t-1, :, :] + pred_vel[:, t, :, :]
                target_positions = targets[..., 0:2] 
                
                predictions_list = []
                targets_list = []
                for b in range(B):
                    valid_idx = (mask[b] == 1).nonzero(as_tuple=False).squeeze()
                    if valid_idx.numel() == 0:
                        continue
                    predictions_list.append(pred_positions[b, :, valid_idx, :])
                    targets_list.append(target_positions[b, :, valid_idx, :])
                if len(predictions_list) > 0:
                    ade_batch = calculate_ade(predictions_list, targets_list)
                    fde_batch = calculate_fde(predictions_list, targets_list)
                else:
                    ade_batch, fde_batch = 0.0, 0.0
                total_ade += ade_batch if isinstance(ade_batch, float) else ade_batch.item()
                total_fde += fde_batch if isinstance(fde_batch, float) else fde_batch.item()
        
        avg_test_loss = total_loss / num_batches
        avg_ade = total_ade / num_batches
        avg_fde = total_fde / num_batches
        print(f"Test Loss (velocity MSE): {avg_test_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")
        return avg_test_loss, avg_ade, avg_fde
        
    def predict(self, obs_list):
        """
        Predict future absolute positions for a batch of samples using the GCN predictor that uses the full past trajectory.
        
        Args:
            obs_list: A list of samples (batch), where each sample is a list of observed positions [x, y].
                      (Each sample corresponds to one agent's trajectory.)
        
        Process:
          - For each sample:
              * Convert the observed trajectory to a tensor of shape (T, 2).
              * Compute velocity differences (i.e. differences between consecutive positions).
              * Pad or truncate the velocity sequence to have length (self.input_len - 1).
              * Extract the last observed position (current position).
          - Stack these into a tensor of shape (B, self.input_len - 1, 2) for velocities and (B, 2) for current positions.
          - Pad the node (agent) dimension from B to self.max_nodes:
                * The velocity tensor becomes (self.max_nodes, self.input_len - 1, 2)
                * The current positions become (self.max_nodes, 2)
          - Add a batch dimension so that:
                obs_vel: (1, self.input_len - 1, self.max_nodes, 2)
                curr_pos: (1, self.max_nodes, 2)
          - Create a mask of shape (1, self.max_nodes) with ones for the first B nodes and zeros for the padded nodes.
          - Optionally, normalize the velocity tensor (normalizing each 2D vector using self.mean[...,2:4] and self.std[...,2:4]).
          - Flatten the time dimension for each node to get features of shape (1, self.max_nodes, (self.input_len - 1)*2).
          - Compute the adjacency matrix using the valid current positions (first B rows) and pad it to shape (self.max_nodes, self.max_nodes); then expand to batch dimension.
          - Forward pass: the model (which expects input shape (1, self.max_nodes, in_features) with in_features=(self.input_len-1)*2) outputs predicted future velocities of shape (1, self.future_length, self.max_nodes, 2).
          - Reconstruct future absolute positions by cumulatively summing the predicted velocities, starting from curr_pos.
          - Filter out padded nodes using the mask and return only the valid predictions.
        
        Returns:
            A NumPy array of shape (B, self.future_length, 2) representing the predicted future absolute positions
            for the valid agents.
        """

        B = len(obs_list)  
        vel_list = []  
        curr_list = [] 
        
        for traj in obs_list:
            traj_tensor = torch.tensor(traj, dtype=torch.float32, device=self.device) 
            if traj_tensor.ndim == 1:
                traj_tensor = traj_tensor.unsqueeze(0)
            T = traj_tensor.shape[0]
            if T > 1:
                vel = traj_tensor[1:] - traj_tensor[:-1]  
            else:
                vel = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            if vel.shape[0] < self.input_len - 1:
                pad_size = self.input_len - 1 - vel.shape[0]
                pad_vel = torch.zeros((pad_size, 2), dtype=torch.float32, device=self.device)
                vel = torch.cat([pad_vel, vel], dim=0)
            elif vel.shape[0] > self.input_len - 1:
                vel = vel[-(self.input_len - 1):]
            vel_list.append(vel)        
            curr_list.append(traj_tensor[-1, :]) 
        
        vel_tensor = torch.stack(vel_list, dim=0)
        curr_tensor = torch.stack(curr_list, dim=0)
        
        if B < self.max_nodes:
            pad_size = self.max_nodes - B
            pad_vel = torch.zeros((pad_size, self.input_len - 1, 2), dtype=torch.float32, device=self.device)
            vel_tensor = torch.cat([vel_tensor, pad_vel], dim=0) 
            pad_curr = torch.zeros((pad_size, 2), dtype=torch.float32, device=self.device)
            curr_tensor = torch.cat([curr_tensor, pad_curr], dim=0) 
            
        obs_vel = vel_tensor.unsqueeze(0)
        curr_pos = curr_tensor.unsqueeze(0)
        mask = torch.cat([torch.ones(B, device=self.device),torch.zeros(self.max_nodes - B, device=self.device)], dim=0)
        mask_tensor = mask.unsqueeze(0)  
    
        if self.normalize and self.mean is not None and self.std is not None:
            B_v, N_v, T_v, C = obs_vel.shape
            obs_vel = obs_vel.view(B_v * T_v * N_v, C)
            obs_vel = (obs_vel - self.mean[..., 2:4]) / self.std[..., 2:4]
            obs_vel = obs_vel.view(B_v, N_v, T_v, C)
            

        B_v, N_v, T_v, C = obs_vel.shape
        in_features = T_v * C 
        feats_flat = obs_vel.reshape(B_v, N_v, in_features)

        adj_valid = compute_adjacency_matrix(curr_pos[0][:B].tolist(), threshold=100, normalize=True)
        adj = torch.zeros(self.max_nodes, self.max_nodes, device=self.device)
        adj[:B, :B] = adj_valid
        adj = adj.unsqueeze(0)
    
        with torch.no_grad():
            pred_vel_norm = self.model(feats_flat, adj, self.future_length)
            if self.normalize and self.mean is not None and self.std is not None:
                B_p, T_pred, N_p, C_p = pred_vel_norm.shape
                pred_vel_norm = pred_vel_norm.view(B_p * T_pred * N_p, C_p)
                pred_vel_norm = pred_vel_norm * self.std[..., 2:4] + self.mean[..., 2:4]
                pred_vel = pred_vel_norm.view(B_p, T_pred, N_p, C_p)
            else:
                pred_vel = pred_vel_norm
    
        pred_positions = torch.zeros((self.future_length, self.max_nodes, 2), device=self.device)
        pred_positions[0] = curr_pos[0] + pred_vel[0, 0]  
        for t in range(1, self.future_length):
            pred_positions[t] = pred_positions[t - 1] + pred_vel[0, t]
        
        # Filter out padded nodes using the mask.
        valid_idx = (mask_tensor[0] == 1).nonzero(as_tuple=False).squeeze()
        if valid_idx.ndim == 0:
            valid_idx = valid_idx.unsqueeze(0)
            
        final_pred = pred_positions[:, valid_idx, :]  
        final_pred = final_pred.permute(1, 0, 2)
        return final_pred.cpu().numpy()


