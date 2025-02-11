#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:47:47 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim

def masked_mse_loss(pred, target, mask):
    """
    pred, target: (B, T, N, out_dim)  (predicted velocities, ground-truth velocities)
    mask:         (B, N) => 1.0 valid node, 0.0 padded

    Returns a scalar masked MSE over valid nodes.
    """
    # Expand mask => (B, 1, N, 1) => broadcast with (B, T, N, out_dim)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # shape (B,1,N,1)
    loss = (pred - target)**2
    loss = loss * mask_expanded
    return loss.sum() / (mask_expanded.sum())

# Optional, if you want to do a masked MSE on velocities only
# but that is handled above.


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x:   (B, N, in_features)  => velocity features
        adj: (B, N, N) => adjacency
        returns: (B, N, out_features)
        """
        support = torch.matmul(x, self.weight)   # (B, N, out_features)
        out     = torch.matmul(adj, support)     # (B, N, out_features)
        if self.bias is not None:
            out += self.bias
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=2):
        super().__init__()
        layers = []
        layers.append(GCNLayer(in_size, hidden_size))
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_size, hidden_size))
        self.gcn_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, feats, adj):
        """
        feats: (B, N, in_size) 
        adj:   (B, N, N)
        returns: (B, N, hidden_size)
        """
        x = feats
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.activation(x)
        return x


class GCNDecoder(nn.Module):
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
        x: (B, N, hidden_size)
        returns: (B, N, out_size)
        """
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, adj)
            if i < len(self.gcn_layers) - 1:
                x = self.activation(x)
        return x


class GCNSeq2Seq(nn.Module):
    """
    Simplified seq2seq: we encode once, then decode future_len times.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, feats, adj, future_len):
        """
        feats: (B, N, in_size)
        adj:   (B, N, N)
        future_len: T_future

        Returns: (B, T_future, N, out_size)
        """
        latent = self.encoder(feats, adj)  # (B, N, hidden_size)
        outs   = []
        for _ in range(future_len):
            decoded = self.decoder(latent, adj)  # (B, N, out_size)
            outs.append(decoded.unsqueeze(1))    # => (B, 1, N, out_size)
        return torch.cat(outs, dim=1)            # => (B, T_future, N, out_size)



class GCNPredictor:
    """
    GCN-based Seq2Seq that trains *only on velocity* from [x,y,vx,vy].
    
    Data assumptions:
      - feats:   (B, N, 4) => [x, y, vx, vy]
      - adj:     (B, N, N)
      - targets: (B, T_future, N, 4) => [x, y, vx, vy] for future
      - mask:    (B, N)

    We slice feats[...,2:4], targets[...,2:4] => shape (B,N,2) / (B,T,N,2),
    feed them to the GCN, compute masked MSE on velocity.
    If self.normalize==True, we do (vel - mean) / std with mean/std => shape (2,).
    """

    def __init__(self, observation_length, future_length, max_nodes, device, normalize, checkpoint_file=None):
        """
        We'll fix in_size=2, out_size=2, hidden_size=64, num_layers=2, etc.
        Because we're ignoring x,y columns entirely.
        
        Args:
          future_length (int): number of future velocity steps
          device (str or torch.device)
          normalize (bool): if True, we do (vx,vy - mean)/(std)
          checkpoint_file (str): optional
        """
        self.future_length = future_length
        self.input_len = observation_length
        self.device = device

        self.normalize = normalize

        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.patience = 5
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # GCN config
        self.hidden_size = 256
        self.num_layers  = 2
        self.in_size     = 2  # only (v_x,v_y)
        self.out_size    = 2  # predicting (v_x,v_y) for the future

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
            self.mean = ckpt.get("mean", None)
            self.std  = ckpt.get("std", None)


    def save_checkpoint(self, folder):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            ckpt["mean"] = self.mean
            ckpt["std"]  = self.std
        torch.save(ckpt, f"{folder}/trained_gcn_model.pth")
        print(f"Checkpoint saved to {folder}/trained_gcn_model.pth")


    def train(self, train_loader, valid_loader=None, ckpt_folder=None):
        """
        Expects each batch to yield:
          feats:   (B, N, 4) => but we only want feats[...,2:4]
          adj:     (B, N, N)
          targets: (B, T, N, 4) => also only want targets[...,2:4]
          mask:    (B, N)
        We'll do MSE on velocity columns => shape(B,T,N,2).
        If self.normalize => mean,std => shape(2,).
        """
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device)  # shape(2,)
            self.std  = train_loader.dataset.std.to(self.device)   # shape(2,)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                feats, adj, targets, mask = batch
                # feats=> (B,N,4), we slice velocity => feats[...,2:4] => shape(B,N,2)
                # targets => (B,T,N,4), slice => (B,T,N,2)

                feats   = feats.to(self.device)
                adj     = adj.to(self.device)
                targets = targets.to(self.device)
                mask    = mask.to(self.device)

                # 1) Extract velocity columns
                feats_vel   = feats[..., 2:4]    # (B,N,2)
                targets_vel = targets[..., 2:4]  # (B,T,N,2)

                self.optimizer.zero_grad()

                # 2) Optionally normalize velocity
                if self.normalize and (self.mean is not None) and (self.std is not None):
                    # feats_vel => shape(B,N,2), we do => (B*N,2) for broadcast
                    B, N, C = feats_vel.shape  # C=2
                    feats_2d = feats_vel.view(B*N, C)
                    feats_2d = (feats_2d - self.mean[2:]) / self.std[2:]
                    feats_vel = feats_2d.view(B, N, C)

                    # targets_vel => shape(B,T,N,2) => reshape => (B*T*N,2)
                    B2, T2, N2, C2 = targets_vel.shape
                    tgt_2d = targets_vel.view(B2*T2*N2, C2)
                    tgt_2d = (tgt_2d - self.mean[2:]) / self.std[2:]
                    targets_vel = tgt_2d.view(B2, T2, N2, C2)

                # 3) Forward => shape(B, T, N, 2)
                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])

                # 4) masked MSE
                loss = self.criterion(pred_vel, targets_vel, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] => Train Loss: {avg_loss:.4f}")

            # Validation
            if valid_loader is not None:
                val_loss = self.validate(valid_loader)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    if ckpt_folder:
                        self.save_checkpoint(ckpt_folder)
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break

        if ckpt_folder and valid_loader is None:
            self.save_checkpoint(ckpt_folder)


    def validate(self, valid_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                feats, adj, targets, mask = batch
                feats   = feats.to(self.device)
                adj     = adj.to(self.device)
                targets = targets.to(self.device)
                mask    = mask.to(self.device)

                # Only velocity
                feats_vel   = feats[..., 2:4]    # shape(B,N,2)
                targets_vel = targets[..., 2:4]  # shape(B,T,N,2)

                if self.normalize and (self.mean is not None) and (self.std is not None):
                    # feats_vel => (B,N,2) => (B*N,2)
                    B, N, C = feats_vel.shape
                    f2d = feats_vel.view(B*N, C)
                    f2d = (f2d - self.mean) / self.std
                    feats_vel = f2d.view(B, N, C)

                    # targets_vel => (B,T,N,2)
                    B2, T2, N2, C2 = targets_vel.shape
                    t2d = targets_vel.view(B2*T2*N2, C2)
                    t2d = (t2d - self.mean) / self.std
                    targets_vel = t2d.view(B2, T2, N2, C2)

                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss


    def evaluate(self, test_loader):
        """
        If you want to reconstruct positions (ADE/FDE), you'd need 
        to store "x,y" somewhere else, but we no longer feed them into the GCN.
        
        For demonstration, let's do velocity MSE or 
        we can do a 'fake' ADE if we know the last positions from feats -- but 
        we actually do not store them in feats anymore. 
        So let's just compute MSE on velocity for demonstration.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(test_loader)

        with torch.no_grad():
            for feats, adj, targets, mask in test_loader:
                feats   = feats.to(self.device)
                adj     = adj.to(self.device)
                targets = targets.to(self.device)
                mask    = mask.to(self.device)

                # velocity only
                feats_vel   = feats[..., 2:4]    # shape(B,N,2)
                targets_vel = targets[..., 2:4]  # shape(B,T,N,2)

                if self.normalize and (self.mean is not None) and (self.std is not None):
                    # feats => normalize
                    B, N, C = feats_vel.shape
                    f2d = feats_vel.view(B*N, C)
                    f2d = (f2d - self.mean) / self.std
                    feats_vel = f2d.view(B, N, C)

                    # targets => normalize
                    B2, T2, N2, C2 = targets_vel.shape
                    t2d = targets_vel.view(B2*T2*N2, C2)
                    t2d = (t2d - self.mean) / self.std
                    targets_vel = t2d.view(B2, T2, N2, C2)

                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()

        avg_test_loss = total_loss / num_batches
        print(f"Test Loss (velocity MSE): {avg_test_loss:.4f}")
        return avg_test_loss


    def predict(self, feats, adj, future_len, mask=None):
        """
        Given (B,N,4) => we only slice velocity => (B,N,2), 
        produce predicted velocities => (B, future_len, N, 2).
        If self.normalize => we do the same (v-mean)/std, then unnormalize.
        """
        self.model.eval()
        feats = feats.to(self.device)
        adj   = adj.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            feats_vel = feats[..., 2:4]  # (B,N,2)

            if self.normalize and self.mean is not None and self.std is not None:
                B, N, C = feats_vel.shape
                f2d = feats_vel.view(B*N, C)
                f2d = (f2d - self.mean) / self.std
                feats_vel = f2d.view(B, N, C)

            pred_vel_norm = self.model(feats_vel, adj, future_len)  # (B,future_len,N,2)
            if self.normalize and self.mean is not None and self.std is not None:
                B2, T2, N2, C2 = pred_vel_norm.shape
                p2d = pred_vel_norm.view(B2*T2*N2, C2)
                p2d = p2d * self.std + self.mean
                pred_vel = p2d.view(B2, T2, N2, C2)
            else:
                pred_vel = pred_vel_norm

        return pred_vel

#####################################################################################################################
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x:   (B, N, in_features)
        adj: (B, N, N) - we will add self-loops so no row is empty
        """
        B, N, _ = x.shape
        # 1) Possibly add self-loops => ensure no row is purely zeros
        #    so e won't become all -inf for a row
        #    Make a copy if you don't want to mutate the input
        #    or define some "force_self_loops" param.
        adj = adj.clone()
        idx = torch.arange(N, device=adj.device)
        adj[:, idx, idx] = 1.0  # add self-loop

        # 2) Linear transform
        h = torch.matmul(x, self.W)  # (B, N, out_features)

        # 3) Compute e_ij => shape (B, N, N)
        #    e_ij = a^T [h_i || h_j]
        h_i = h.unsqueeze(2)  # => (B, N, 1, out_features)
        h_j = h.unsqueeze(1)  # => (B, 1, N, out_features)
        h_cat = torch.cat([h_i.repeat(1,1,N,1), h_j.repeat(1,N,1,1)], dim=-1)  
        e = torch.matmul(h_cat, self.a).squeeze(-1)  # => (B, N, N)

        # 4) mask out no-edge => set -1e9
        zero_mask = (adj < 1e-9)  # adjacency=0
        e = e.masked_fill(zero_mask, -1e9)

        # 5) softmax over dim=-1
        alpha = torch.softmax(e, dim=-1)  # => shape (B, N, N)

        # 6) Weighted sum
        alpha_exp = alpha.unsqueeze(-1)   # (B, N, N, 1)
        # h_j => shape (B, 1, N, out_features), repeat => (B, N, N, out_features)
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1) 
        out = (alpha_exp * h_j).sum(dim=2)  # => (B,N,out_features)

        if self.bias is not None:
            out += self.bias

        return out



class GATEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=2):
        super().__init__()
        layers = []
        layers.append(GATLayer(in_size, hidden_size))
        for _ in range(num_layers - 1):
            layers.append(GATLayer(hidden_size, hidden_size))
        self.gat_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, feats, adj):
        """
        feats: (B, N, in_size)
        adj:   (B, N, N)
        returns: (B, N, hidden_size)
        """
        x = feats
        for layer in self.gat_layers:
            x = layer(x, adj)
            x = self.activation(x)
        return x


class GATDecoder(nn.Module):
    def __init__(self, hidden_size, out_size, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(GATLayer(hidden_size, hidden_size))
        layers.append(GATLayer(hidden_size, out_size))
        self.gat_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        # x: (B, N, hidden_size)
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, adj)
            if i < len(self.gat_layers)-1:
                x = self.activation(x)
        return x


class GATSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, feats, adj, future_len):
        """
        feats: (B, N, in_size)
        adj:   (B, N, N)
        future_len: T_future
        returns: (B, T_future, N, out_size)
        """
        latent = self.encoder(feats, adj)  # (B, N, hidden_size)
        outs   = []
        for _ in range(future_len):
            decoded = self.decoder(latent, adj)  # (B, N, out_size)
            outs.append(decoded.unsqueeze(1))    # => (B,1,N,out_size)
        return torch.cat(outs, dim=1)            # => (B,T_future,N,out_size)




class GATPredictor:
    """
    GAT-based Seq2Seq that trains only on velocity from [x,y,vx,vy].
    
    Data assumptions:
      - feats:   (B, N, 4) => [x,y,vx,vy]
      - adj:     (B, N, N)
      - targets: (B, T_future, N, 4)
      - mask:    (B, N)
    
    We slice feats[...,2:4] => shape(B,N,2), 
    slice targets[...,2:4] => shape(B,T,N,2).
    We'll do a masked MSE on velocity columns.
    If self.normalize => we do partial velocity normalization => shape(2,).
    """

    def __init__(self, observation_length, future_length, max_nodes, device, normalize, checkpoint_file=None):
        """
        We'll fix in_size=2, out_size=2, hidden_size=64, num_layers=2, etc.
        Because we ignore x,y columns entirely.
        
        If you want more heads, you'll adapt GATLayer to do multi-head.
        """
        self.future_length = future_length
        self.device = device
        self.normalize = normalize

        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.patience = 5
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # GAT config
        self.hidden_size = 256
        self.num_layers  = 2
        self.in_size     = 2  # only velocity (vx,vy)
        self.out_size    = 2  # predict (vx,vy)
        
        encoder = GATEncoder(self.in_size, self.hidden_size, self.num_layers)
        decoder = GATDecoder(self.hidden_size, self.out_size, self.num_layers)
        self.model = GATSeq2Seq(encoder, decoder).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = masked_mse_loss
        
        if checkpoint_file is not None:
            print(f"Loading GAT from {checkpoint_file}")
            ckpt = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.mean = ckpt.get("mean", None)
            self.std  = ckpt.get("std", None)

    def save_checkpoint(self, folder):
        dct = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            dct["mean"] = self.mean
            dct["std"]  = self.std
        torch.save(dct, f"{folder}/trained_gat_model.pth")
        print(f"Checkpoint saved to {folder}/trained_gat_model.pth")

    def train(self, train_loader, valid_loader=None, ckpt_folder=None):
        """
        Expects:
          feats:   (B, N, 4)
          adj:     (B, N, N)
          targets: (B, T, N, 4)
          mask:    (B, N)
        We'll do MSE on velocity columns => shape(B,T,N,2).
        If self.normalize => mean,std => shape(2,).
        """
        if self.normalize:
            # mean, std => shape(2,)
            self.mean = train_loader.dataset.mean.to(self.device)
            self.std  = train_loader.dataset.std.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for (feats, adj, targets, mask) in train_loader:
                feats   = feats.to(self.device)   # (B,N,4)
                adj     = adj.to(self.device)     # (B,N,N)
                targets = targets.to(self.device) # (B,T,N,4)
                mask    = mask.to(self.device)    # (B,N)

                feats_vel   = feats[..., 2:4]    # (B,N,2)
                targets_vel = targets[..., 2:4]  # (B,T,N,2)

                self.optimizer.zero_grad()

                # partial velocity normalization
                if self.normalize and self.mean is not None and self.std is not None:
                    B, N, C = feats_vel.shape  # (B,N,2)
                    fv_2d = feats_vel.view(B*N, C)
                    fv_2d = (fv_2d - self.mean[2:]) / self.std[2:]
                    feats_vel = fv_2d.view(B, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    tv_2d = targets_vel.view(B2*T2*N2, C2)
                    tv_2d = (tv_2d - self.mean[2:]) / self.std[2:]
                    targets_vel = tv_2d.view(B2, T2, N2, C2)

                # forward => shape(B,T2,N,2)
                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
                
                loss = self.criterion(pred_vel, targets_vel, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] => Train Loss: {avg_loss:.4f}")

            if valid_loader is not None:
                val_loss = self.validate(valid_loader)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    if ckpt_folder:
                        self.save_checkpoint(ckpt_folder)
                else:
                    self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        if ckpt_folder and (valid_loader is None):
            self.save_checkpoint(ckpt_folder)

    def validate(self, valid_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for feats, adj, targets, mask in valid_loader:
                feats   = feats.to(self.device)
                adj     = adj.to(self.device)
                targets = targets.to(self.device)
                mask    = mask.to(self.device)

                feats_vel   = feats[..., 2:4]
                targets_vel = targets[..., 2:4]

                if self.normalize and self.mean is not None and self.std is not None:
                    B, N, C = feats_vel.shape
                    fv_2d = feats_vel.view(B*N, C)
                    fv_2d = (fv_2d - self.mean) / self.std
                    feats_vel = fv_2d.view(B, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    tv_2d = targets_vel.view(B2*T2*N2, C2)
                    tv_2d = (tv_2d - self.mean) / self.std
                    targets_vel = tv_2d.view(B2, T2, N2, C2)

                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate(self, test_loader):
        """
        We'll do velocity MSE. If you want ADE/FDE on positions, you'd 
        need the x,y somewhere else. This is purely velocity-based.
        """
        self.model.eval()
        total_loss = 0.0
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
                    B, N, C = feats_vel.shape
                    fv_2d = feats_vel.view(B*N, C)
                    fv_2d = (fv_2d - self.mean) / self.std
                    feats_vel = fv_2d.view(B, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    tv_2d = targets_vel.view(B2*T2*N2, C2)
                    tv_2d = (tv_2d - self.mean) / self.std
                    targets_vel = tv_2d.view(B2, T2, N2, C2)

                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()

        avg_test_loss = total_loss / num_batches
        print(f"Test Loss (velocity MSE): {avg_test_loss:.4f}")
        return avg_test_loss

    def predict(self, feats, adj, future_len, mask=None):
        """
        Given (B,N,4) => we only slice velocity => (B,N,2), 
        produce predicted velocities => (B, future_len, N, 2).
        If self.normalize => we do the same (v-mean)/std, then unnormalize.
        """
        self.model.eval()
        feats = feats.to(self.device)
        adj   = adj.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            feats_vel = feats[..., 2:4]

            if self.normalize and self.mean is not None and self.std is not None:
                B, N, C = feats_vel.shape
                fv_2d = feats_vel.view(B*N, C)
                fv_2d = (fv_2d - self.mean) / self.std
                feats_vel = fv_2d.view(B, N, C)

            pred_vel_norm = self.model(feats_vel, adj, future_len)
            if self.normalize and self.mean is not None and self.std is not None:
                B2, T2, N2, C2 = pred_vel_norm.shape
                p2d = pred_vel_norm.view(B2*T2*N2, C2)
                p2d = p2d * self.std + self.mean
                pred_vel = p2d.view(B2, T2, N2, C2)
            else:
                pred_vel = pred_vel_norm

        return pred_vel