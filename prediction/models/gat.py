#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [DATE]
Author: [Your Name]

GAT-based Seq2Seq Predictor using a single-parameter attention mechanism.
The model predicts future velocities using only [vx, vy] channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from evaluation.distance_metrics import calculate_ade, calculate_fde
from dataloaders.frame_loader import compute_adjacency_matrix



def masked_mse_loss(pred, target, mask):
    """
    Compute the Mean Squared Error loss over valid nodes only.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, T, N, out_dim).
        target (torch.Tensor): Ground-truth tensor of shape (B, T, N, out_dim).
        mask (torch.Tensor): Binary mask tensor of shape (B, N).

    Returns:
        torch.Tensor: Averaged MSE loss.
    """
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
    loss = (pred - target) ** 2
    loss = loss * mask_expanded
    return loss.sum() / mask_expanded.sum()


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer using a single parameter for the attention mechanism.

    This layer computes attention coefficients as:
        e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
    where 'a' is a single learnable parameter applied on the concatenated features.

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        dropout (float): Dropout probability.
        alpha (float): Negative slope for the LeakyReLU.
        concat (bool): Whether to apply an ELU after aggregation.
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Node features of shape (B, N, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (B, N, N).

        Returns:
            torch.Tensor: Updated node features of shape (B, N, out_features).
        """
        # Apply linear transformation
        Wh = self.W(x) 
        B, N, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1) 
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)   

        # Compute attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax normalization and dropout on the attention scores
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate neighbor features weighted by attention coefficients
        h_prime = torch.matmul(attention, Wh)  
        return F.elu(h_prime) if self.concat else h_prime


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer where each head uses the single-parameter approach.
    
    Args:
        in_features (int): Number of input features per node.
        out_features (int): Output features per head.
        dropout (float): Dropout probability.
        alpha (float): Negative slope for LeakyReLU.
        nheads (int): Number of attention heads.
        concat (bool): Whether to concatenate the outputs of each head.
    """
    def __init__(self, in_features, out_features, dropout, alpha, nheads, concat=True):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=concat)
            for _ in range(nheads)
        ])
        self.concat = concat

    def forward(self, x, adj):
        if self.concat:
            return torch.cat([head(x, adj) for head in self.heads], dim=-1)
        else:
            return torch.mean(torch.stack([head(x, adj) for head in self.heads]), dim=0)


class GATEncoder(nn.Module):
    """
    GAT Encoder composed of stacked multi-head GAT layers.
    
    Args:
        in_features (int): Number of input features per node.
        hidden_size (int): Hidden size per head.
        dropout (float): Dropout probability.
        alpha (float): Negative slope for LeakyReLU.
        nheads (int): Number of heads in the first layer.
        num_layers (int): Number of GAT layers.
    """
    def __init__(self, in_features, hidden_size, dropout, alpha, nheads, num_layers=2):
        super().__init__()
        layers = []

        layers.append(MultiHeadGATLayer(in_features, hidden_size, dropout, alpha, nheads, concat=True))
        for _ in range(num_layers - 2):
            layers.append(MultiHeadGATLayer(hidden_size * nheads, hidden_size, dropout, alpha, nheads, concat=True))

        if num_layers > 1:
            layers.append(MultiHeadGATLayer(hidden_size * nheads, hidden_size, dropout, alpha, nheads=1, concat=False))
        self.gat_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, adj)
            if i < len(self.gat_layers) - 1:
                x = self.activation(x)
        return x


class GATDecoder(nn.Module):
    """
    GAT Decoder that uses stacked GAT layers to generate output features.
    
    Args:
        hidden_size (int): Input feature size from the encoder.
        out_size (int): Desired output feature size.
        dropout (float): Dropout probability.
        alpha (float): Negative slope for LeakyReLU.
        num_layers (int): Number of GAT layers.
    """
    def __init__(self, hidden_size, out_size, dropout, alpha, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(GraphAttentionLayer(hidden_size, hidden_size, dropout, alpha, concat=True))
        layers.append(GraphAttentionLayer(hidden_size, out_size, dropout, alpha, concat=False))
        self.gat_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, adj)
            if i < len(self.gat_layers) - 1:
                x = self.activation(x)
        return x


class GATSeq2Seq(nn.Module):
    """
    GAT-based Seq2Seq model that integrates an encoder and a decoder.
    
    Args:
        encoder (nn.Module): The GAT encoder.
        decoder (nn.Module): The GAT decoder.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, feats, adj, future_len):
        """
        Args:
            feats (torch.Tensor): Input features of shape (B, N, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (B, N, N).
            future_len (int): Number of future time steps to predict.

        Returns:
            torch.Tensor: Predictions of shape (B, future_len, N, out_size).
        """
        latent = self.encoder(feats, adj)
        outs = []
        for _ in range(future_len):
            decoded = self.decoder(latent, adj)
            outs.append(decoded.unsqueeze(1))
        return torch.cat(outs, dim=1)


class GATPredictor:
    """
    GAT-based Seq2Seq predictor for trajectory prediction.

    Data assumptions:
      - feats:   (B, T_obs, N, 4) where each feature is [x, y, vx, vy].
      - adj:     (B, N, N)
      - targets: (B, T_future, N, 4) where each target is [x, y, vx, vy].
      - mask:    (B, N)

    The model predicts future velocities using only [vx, vy]. During evaluation (ADE/FDE),
    these predicted velocities are integrated into positions using the last observed positions.
    This version uses a flattened past trajectory (i.e. all observed velocity differences)
    for training.
    """
    def __init__(self, observation_length, future_length, device, max_nodes=50, normalize=False, checkpoint_file=None):
        self.future_length = future_length
        self.input_len = observation_length  # T_obs (number of observed positions)
        self.device = device
        self.normalize = normalize
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.patience = 50
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.hidden_size = 256
        self.num_layers = 2
        self.max_nodes = max_nodes

        self.in_size = (observation_length - 1) * 2
        self.out_size = 2
        self.nheads = 2
        self.dropout = 0
        self.alpha = 0.2

        encoder = GATEncoder(self.in_size, self.hidden_size, self.dropout, self.alpha, self.nheads, num_layers=self.num_layers)
        decoder = GATDecoder(self.hidden_size, self.out_size, self.dropout, self.alpha, num_layers=self.num_layers)
        self.model = GATSeq2Seq(encoder, decoder).to(self.device)
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
        Save the model and optimizer state.
        """
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            ckpt["mean"] = self.mean
            ckpt["std"] = self.std
        ckpt["max_nodes"] = self.max_nodes
        torch.save(ckpt, f"{folder}/gat_trained_model.pth")
        print(f"Checkpoint saved to {folder}/gat_lstm_trained_model.pth")

    def train(self, train_loader, valid_loader=None, save_path=None):
        """
        Train the GAT predictor using the flattened past trajectory.
        """
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device)
            self.std = train_loader.dataset.std.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_ade = 0.0
            epoch_fde = 0.0

            load_train = tqdm(train_loader, desc=f"Epoch: {epoch+1}/{self.num_epochs}", leave=False)
            for id_b, batch in enumerate(load_train):
                feats, adj, targets, mask = batch
                feats = feats.to(self.device)
                adj = adj.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                
                feats_vel = feats[..., 2:4]    
                targets_vel = targets[..., 2:4] 
                self.optimizer.zero_grad()

                if self.normalize and (self.mean is not None) and (self.std is not None):
                    B, T_obs, N, C = feats_vel.shape
                    feats_2d = feats_vel.view(B * T_obs * N, C)
                    feats_2d = (feats_2d - self.mean[2:]) / self.std[2:]
                    feats_vel = feats_2d.view(B, T_obs, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    tgt_2d = targets_vel.view(B2 * T2 * N2, C2)
                    tgt_2d = (tgt_2d - self.mean[2:]) / self.std[2:]
                    targets_vel = tgt_2d.view(B2, T2, N2, C2)

                B, T_obs, N, C = feats_vel.shape
                flattened_dim = T_obs * C
                feats_vel_flat = feats_vel.permute(0, 2, 1, 3).reshape(B, N, flattened_dim)

                pred_vel = self.model(feats_vel_flat, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                with torch.no_grad():
                    if self.normalize:
                        pred_vel_un = pred_vel * self.std[2:] + self.mean[2:]
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
            avg_epoch_ade = epoch_ade / len(train_loader)
            avg_epoch_fde = epoch_fde / len(train_loader)
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
        Validate the model on a validation dataset using flattened past trajectory.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for feats, adj, targets, mask in valid_loader:
                feats = feats.to(self.device)
                adj = adj.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                feats_vel = feats[..., 2:4]
                targets_vel = targets[..., 2:4]

                if self.normalize and (self.mean is not None) and (self.std is not None):
                    B, T_obs, N, C = feats_vel.shape
                    feats_2d = feats_vel.view(B * T_obs * N, C)
                    feats_2d = (feats_2d - self.mean[2:]) / self.std[2:]
                    feats_vel = feats_2d.view(B, T_obs, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    t2d = targets_vel.view(B2 * T2 * N2, C2)
                    t2d = (t2d - self.mean[2:]) / self.std[2:]
                    targets_vel = t2d.view(B2, T2, N2, C2)

                B, T_obs, N, C = feats_vel.shape
                flattened_dim = T_obs * C
                feats_vel_flat = feats_vel.permute(0, 2, 1, 3).reshape(B, N, flattened_dim)

                pred_vel = self.model(feats_vel_flat, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()
        avg_val_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate(self, test_loader):
        """
        Evaluate the model on a test dataset using flattened past trajectory.
        """
        self.model.eval()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = len(test_loader)
        with torch.no_grad():
            for feats, adj, targets, mask in test_loader:
                feats = feats.to(self.device)
                adj = adj.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                feats_vel = feats[..., 2:4]
                targets_vel = targets[..., 2:4]

                if self.normalize and (self.mean is not None) and (self.std is not None):
                    B, T_obs, N, C = feats_vel.shape
                    feats_2d = feats_vel.view(B * T_obs * N, C)
                    feats_2d = (feats_2d - self.mean[2:]) / self.std[2:]
                    feats_vel = feats_2d.view(B, T_obs, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    t2d = targets_vel.view(B2 * T2 * N2, C2)
                    t2d = (t2d - self.mean[2:]) / self.std[2:]
                    targets_vel = t2d.view(B2, T2, N2, C2)

                B, T_obs, N, C = feats_vel.shape
                flattened_dim = T_obs * C
                feats_vel_flat = feats_vel.permute(0, 2, 1, 3).reshape(B, N, flattened_dim)

                pred_vel = self.model(feats_vel_flat, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()

                if self.normalize and self.mean is not None and self.std is not None:
                    pred_vel = pred_vel * self.std[2:] + self.mean[2:]
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
        Predict future absolute positions for a batch of samples using the GAT predictor with
        a flattened past trajectory. The procedure is as follows:

          - For each sample (agent trajectory):
              * Convert the observed trajectory (list of [x, y]) into a tensor.
              * Compute velocity differences (i.e. traj[1:] - traj[:-1]).
              * Pad (or truncate) the velocity sequence to have length (self.input_len - 1).
              * Extract the last observed position (current position).
          - Stack these to form:
                * A velocity tensor of shape (B, self.input_len - 1, 2)
                * A current position tensor of shape (B, 2)
          - Pad the node dimension to self.max_nodes.
          - Add a batch dimension so that:
                obs_vel:  (1, self.max_nodes, self.input_len - 1, 2)
                curr_pos: (1, self.max_nodes, 2)
          - Create a mask indicating valid nodes.
          - (Optionally) Normalize the velocity tensor.
          - Flatten the time dimension for each node to obtain features of shape (1, self.max_nodes, (self.input_len - 1)*2).
          - Compute the adjacency matrix using the valid current positions, then pad it.
          - Forward pass through the model to predict future velocities.
          - Reconstruct future positions by cumulatively summing velocities starting from curr_pos.
          - Filter out padded nodes and return the predictions as a NumPy array.
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
        mask = torch.cat([torch.ones(B, device=self.device), torch.zeros(self.max_nodes - B, device=self.device)], dim=0)
        mask_tensor = mask.unsqueeze(0)

        if self.normalize and self.mean is not None and self.std is not None:
            B_v, N_v, T_v, C = obs_vel.shape
            obs_vel = obs_vel.view(B_v * T_v * N_v, C)
            obs_vel = (obs_vel - self.mean[2:]) / self.std[2:]
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
                pred_vel_norm = pred_vel_norm * self.std[2:] + self.mean[2:]
                pred_vel = pred_vel_norm.view(B_p, T_pred, N_p, C_p)
            else:
                pred_vel = pred_vel_norm

        pred_positions = torch.zeros((self.future_length, self.max_nodes, 2), device=self.device)
        pred_positions[0] = curr_pos[0] + pred_vel[0, 0]
        for t in range(1, self.future_length):
            pred_positions[t] = pred_positions[t - 1] + pred_vel[0, t]


        valid_idx = (mask_tensor[0] == 1).nonzero(as_tuple=False).squeeze()
        if valid_idx.ndim == 0:
            valid_idx = valid_idx.unsqueeze(0)

        final_pred = pred_positions[:, valid_idx, :] 
        final_pred = final_pred.permute(1, 0, 2)         
        return final_pred.cpu().numpy()
