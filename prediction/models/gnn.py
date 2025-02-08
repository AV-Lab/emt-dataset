#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:47:47 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Example simple loss function that applies a mask.
def masked_mse_loss(outputs, targets, mask):
    """
    Compute MSE loss for valid nodes only.
    
    Args:
        outputs: Tensor of shape (batch_size, T_future, max_nodes, output_size)
        targets: Tensor of shape (batch_size, T_future, max_nodes, output_size)
        mask:    Tensor of shape (batch_size, max_nodes)
    Returns:
        A scalar loss value.
    """
    # Expand mask to match outputs: (batch_size, 1, max_nodes, 1)
    #print("OUTPUTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS", outputs)
    #print("TRAGETSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS", targets)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
    loss = (outputs - targets) ** 2
    loss = loss * mask_expanded  # Only consider valid nodes
    
    # Average loss over the valid nodes
    return loss.sum() / mask_expanded.sum()

class GCNPredictor:
    """
    A GCN-based trajectory prediction model that supports node masking for padded nodes.
    
    Node features are assumed to have dimension 2*(T_past+1) (flattened past trajectory plus current location).
    """
    
    class GCNLayer(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x, adj):
            support = torch.matmul(x, self.weight)  # (batch_size, num_nodes, out_features)
            out = torch.matmul(adj, support)        # (batch_size, num_nodes, out_features)
            if self.bias is not None:
                out += self.bias
            return out

    class GCNEncoder(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=2):
            super().__init__()
            self.gcn_layers = nn.ModuleList()
            self.gcn_layers.append(GCNPredictor.GCNLayer(input_size, hidden_size))
            for _ in range(num_layers - 1):
                self.gcn_layers.append(GCNPredictor.GCNLayer(hidden_size, hidden_size))
            self.activation = nn.ReLU()

        def forward(self, x, adj):
            for layer in self.gcn_layers:
                x = layer(x, adj)
                x = self.activation(x)
            return x

    class GCNDecoder(nn.Module):
        def __init__(self, hidden_size, output_size, num_layers=2):
            super().__init__()
            self.gcn_layers = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gcn_layers.append(GCNPredictor.GCNLayer(hidden_size, hidden_size))
            self.gcn_layers.append(GCNPredictor.GCNLayer(hidden_size, output_size))
            self.activation = nn.ReLU()

        def forward(self, x, adj):
            for i, layer in enumerate(self.gcn_layers):
                x = layer(x, adj)
                if i < len(self.gcn_layers) - 1:
                    x = self.activation(x)
            return x

    class GCNSeq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, source_feats, source_adj, target_len):
            batch_size = source_feats.size(0)
            latent = self.encoder(source_feats, source_adj)
            outputs = []
            for _ in range(target_len):
                decoded = self.decoder(latent, source_adj)
                outputs.append(decoded.unsqueeze(1))
            return torch.cat(outputs, dim=1)  # (batch_size, target_len, num_nodes, output_size)

    def __init__(self, past_length, future_length, max_nodes, checkpoint_file=None):
        """
        Args:
            max_nodes (int): Fixed maximum number of nodes.
            past_length (int): Number of past steps (T_past).
            future_length (int): Number of future steps (T_future).
            checkpoint_file (str, optional): Path to a saved model checkpoint.
        """
        # Feature size is 2*(T_past+1): past trajectory (flattened) + current location.
        self.input_size = 2 * past_length
        self.output_size = 2
        self.hidden_size = 64
        self.num_layers = 2

        self.max_nodes = max_nodes
        self.future_length = future_length

        self.num_epochs = 200
        self.learning_rate = 0.001
        self.patience = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = self.GCNEncoder(self.input_size, self.hidden_size, self.num_layers)
        decoder = self.GCNDecoder(self.hidden_size, self.output_size, self.num_layers)
        self.model = self.GCNSeq2Seq(encoder, decoder).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        if checkpoint_file is not None:
            print("Loading weights from checkpoint:", checkpoint_file)
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_checkpoint(self, save_dir):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{save_dir}/trained_gcn_model.pth")
        print(f"Checkpoint saved to: {save_dir}/trained_gcn_model.pth")

    def train(self, train_loader, valid_loader=None, save_dir=None):
        print(f"Starting training for {self.num_epochs} epochs.")
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for inputs, adjacency, targets, mask in train_loader:
                inputs = inputs.to(self.device)
                adjacency = adjacency.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs, adjacency, self.future_length)
                loss = masked_mse_loss(outputs, targets, mask)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Train Loss: {avg_train_loss:.4f}")

            if valid_loader is not None:
                val_loss = self.validate(valid_loader)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    if save_dir is not None:
                        self.save_checkpoint(save_dir)
                else:
                    self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break

        if save_dir is not None:
            self.save_checkpoint(save_dir)

    def validate(self, valid_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, adjacency, targets, mask in valid_loader:
                inputs = inputs.to(self.device)
                adjacency = adjacency.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                outputs = self.model(inputs, adjacency, self.future_length)
                total_loss += masked_mse_loss(outputs, targets, mask).item()
        avg_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, adjacency, targets, mask in test_loader:
                inputs = inputs.to(self.device)
                adjacency = adjacency.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                outputs = self.model(inputs, adjacency, self.future_length)
                total_loss += masked_mse_loss(outputs, targets, mask).item()
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, input_feats, adjacency, mask, future_length=None):
        self.model.eval()
        if future_length is None:
            future_length = self.future_length
        with torch.no_grad():
            input_feats = input_feats.to(self.device)
            adjacency = adjacency.to(self.device)
            preds = self.model(input_feats, adjacency, future_length)
        # Optionally, you could use the mask here to filter predictions.
        return preds



class GATPredictor:
    def __init__(self, observation_length, prediction_horizon, checkpoint_file=None):
        pass
