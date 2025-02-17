import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluation.distance_metrics import calculate_ade, calculate_fde

def masked_mse_loss(pred, target, mask):
    """
    Compute the Mean Squared Error only on valid nodes.
    
    Args:
      pred, target: (B, T, N, out_dim)  (predicted velocities, ground-truth velocities)
      mask:         (B, N)  with 1.0 for valid nodes, 0.0 for padded
      
    Returns:
      Scalar loss averaged over valid elements.
    """
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1) 
    loss = (pred - target) ** 2
    loss = loss * mask_expanded
    return loss.sum() / mask_expanded.sum()


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
        support = torch.matmul(x, self.weight)  
        out = torch.matmul(adj, support)       
        if self.bias is not None:
            out += self.bias
        return out

class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers=2):
        super().__init__()
        layers = [GCNLayer(in_features, hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_size, hidden_size))
        self.gcn_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
    
    def forward(self, x, adj):
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.activation(x)
        return x 

class GCN_LSTM_Predictor(nn.Module):
    """
    Combined GCN + LSTM model.
    
    For each observation time step, the GCN extracts a spatial feature per agent.
    Then, an LSTM encoder processes the temporal sequence (over T_obs) for each agent.
    An autoregressive LSTM decoder (initialized with the encoder’s final state)
    predicts future velocities.
    
    Input:
      feats: (B, T_obs, N, in_features)   (e.g., in_features=2 for [vx, vy])
      adj:   (B, N, N)  – assumed constant over time.
    
    Output:
      predictions: (B, future_len, N, in_features)
    """
    def __init__(self, in_features, gcn_hidden, lstm_hidden, num_gcn_layers, num_lstm_encoder_layers, num_lstm_decoder_layers, future_len):
        super().__init__()
        self.future_len = future_len
        self.gcn_encoder = GCNEncoder(in_features, gcn_hidden, num_gcn_layers)
        self.lstm_encoder = nn.LSTM(gcn_hidden, lstm_hidden, num_lstm_encoder_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(in_features, lstm_hidden, num_lstm_decoder_layers, batch_first=True)
        self.decoder_fc = nn.Linear(lstm_hidden, in_features)
    
    def forward(self, feats, adj, future_len=None):
        if future_len is None:
            future_len = self.future_len
        B, T_obs, N, F = feats.shape
        
        gcn_outputs = []
        for t in range(T_obs):
            x_t = feats[:, t, :, :]        
            gcn_out = self.gcn_encoder(x_t, adj) 
            gcn_outputs.append(gcn_out.unsqueeze(1))  
        gcn_seq = torch.cat(gcn_outputs, dim=1)
        

        gcn_seq = gcn_seq.transpose(1, 2).contiguous().view(B * N, T_obs, -1)
        
        # LSTM encoder over the temporal sequence.
        _, (h_n, c_n) = self.lstm_encoder(gcn_seq)
        h_last = h_n[-1] 
        c_last = c_n[-1] 
        
        # Initialize decoder state with encoder’s final state.
        decoder_hidden = (
            h_last.unsqueeze(0).repeat(self.lstm_decoder.num_layers, 1, 1),
            c_last.unsqueeze(0).repeat(self.lstm_decoder.num_layers, 1, 1)
        )
        
        # Autoregressive decoding: start with zero vector for each agent.
        decoder_input = torch.zeros(B * N, 1, F, device=feats.device)
        predictions = []
        for _ in range(future_len):
            out, decoder_hidden = self.lstm_decoder(decoder_input, decoder_hidden)
            pred = self.decoder_fc(out)  
            predictions.append(pred)
            decoder_input = pred

        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.view(B, N, future_len, F).transpose(1, 2).contiguous()
        return predictions


class GCNLSTMPredictor:
    """
    GCN-based Seq2Seq predictor that trains only on velocity from [x,y,vx,vy].
    
    Data assumptions:
      - feats:   (B, T_obs, N, 4)  -> [x, y, vx, vy]
      - adj:     (B, N, N)
      - targets: (B, T_future, N, 4) -> [x, y, vx, vy] for future steps
      - mask:    (B, N)
    
    The model uses only the velocity channels (2:4) as input and computes a masked MSE loss
    on velocities. For evaluation (ADE/FDE), the predicted velocities are integrated into
    absolute positions using the last observed positions (x,y from feats[...,0:2]).
    
    Normalization (if enabled) is applied only on the velocity channels.
    """
    def __init__(self, observation_length, future_length, max_nodes, device, normalize, checkpoint_file=None):
        """
        Args:
          observation_length (int): number of observed time steps (T_obs)
          future_length (int): number of future velocity steps (T_future)
          max_nodes (int): maximum number of agents
          device (str or torch.device)
          normalize (bool): if True, apply normalization to (vx, vy)
          checkpoint_file (str): optional checkpoint to load
        """
        self.future_length = future_length
        self.input_len = observation_length
        self.device = device
        self.normalize = normalize

        self.num_epochs = 50
        self.learning_rate = 1e-5
        self.patience = 5
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        self.gcn_hidden_size = 256
        self.lstm_hidden_size = 128
        self.gcn_num_layers  = 2
        self.lstm_encode_num_layers = 2
        self.lstm_decode_num_layers = 2
        self.in_size     = 2   
        self.out_size    = 2

        self.model = GCN_LSTM_Predictor(
            in_features=self.in_size,
            gcn_hidden=self.gcn_hidden_size,
            lstm_hidden=self.lstm_hidden_size,
            num_gcn_layers=self.gcn_num_layers,
            num_lstm_encoder_layers=self.lstm_encode_num_layers,
            num_lstm_decoder_layers=self.lstm_decode_num_layers,
            future_len=self.future_length
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-7)
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

    def train(self, train_loader, valid_loader=None, saving_checkpoint_path=None):
        """
        Expects each batch to yield:
          feats:   (B, T_obs, N, 4)  -> if T_obs is missing, add one.
          adj:     (B, N, N)
          targets: (B, T_future, N, 4) -> we use only targets[...,2:4]
          mask:    (B, N)
        """
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
                feats, adj, targets, mask = batch
                feats   = feats.to(self.device)
                adj     = adj.to(self.device)
                targets = targets.to(self.device)
                mask    = mask.to(self.device)

                feats_vel   = feats[..., 2:4]    
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

                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
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

                    epoch_ade += ade_batch
                    epoch_fde += fde_batch

                load_train.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ADE':  f"{ade_batch:.4f}",
                    'FDE':  f"{fde_batch:.4f}"
                })

                if valid_loader is not None:
                    val_loss = self.validate(valid_loader)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        if saving_checkpoint_path is not None:
                            self.save_checkpoint(saving_checkpoint_path)
                    else:
                        self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        print("Early stopping triggered.")
                        break

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_ade  = epoch_ade  / len(train_loader)
            avg_epoch_fde  = epoch_fde  / len(train_loader)
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train - Loss: {avg_epoch_loss:.4f}, ADE: {avg_epoch_ade:.4f}, FDE: {avg_epoch_fde:.4f}")

        if saving_checkpoint_path is not None and valid_loader is None:
            self.save_checkpoint(saving_checkpoint_path)

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
                
                feats_vel   = feats[..., 2:4]   
                targets_vel = targets[..., 2:4] 

                if self.normalize and (self.mean is not None) and (self.std is not None):
                    B, T_obs, N, C = feats_vel.shape
                    f2d = feats_vel.view(B * T_obs * N, C)
                    f2d = (f2d - self.mean[2:]) / self.std[2:]
                    feats_vel = f2d.view(B, T_obs, N, C)

                    B2, T2, N2, C2 = targets_vel.shape
                    t2d = targets_vel.view(B2 * T2 * N2, C2)
                    t2d = (t2d - self.mean[2:]) / self.std[2:]
                    targets_vel = t2d.view(B2, T2, N2, C2)

                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
                loss = self.criterion(pred_vel, targets_vel, mask)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate(self, test_loader):
        """
        Evaluate the model on a test dataset and compute the velocity MSE loss, ADE, and FDE.
    
        Args:
            test_loader (DataLoader): Data loader yielding (feats, adj, targets, mask).
    
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
                
                if self.normalize and (self.mean is not None) and (self.std is not None):
                    B, T_obs, N, C = feats_vel.shape
                    feats_2d = feats_vel.view(B * T_obs * N, C)
                    feats_2d = (feats_2d - self.mean[2:]) / self.std[2:]
                    feats_vel = feats_2d.view(B, T_obs, N, C)
    
                    B2, T2, N2, C2 = targets_vel.shape
                    t2d = targets_vel.view(B2 * T2 * N2, C2)
                    t2d = (t2d - self.mean[2:]) / self.std[2:]
                    targets_vel = t2d.view(B2, T2, N2, C2)
                
                pred_vel = self.model(feats_vel, adj, targets_vel.shape[1])
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
    
                total_ade += ade_batch
                total_fde += fde_batch
    
        avg_test_loss = total_loss / num_batches
        avg_ade = total_ade / num_batches
        avg_fde = total_fde / num_batches
        print(f"Test Loss (velocity MSE): {avg_test_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")
        return avg_test_loss, avg_ade, avg_fde


    def predict(self, feats, adj, future_len, mask=None):
        """
        Given raw features (B, T_obs, N, 4), use only the velocity channels (2:4) 
        to predict future velocities (B, future_len, N, 2). If normalization is enabled,
        apply it (and then unnormalize).
        """
        self.model.eval()
        feats = feats.to(self.device)
        adj   = adj.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if feats.dim() == 3:
            feats = feats.unsqueeze(1)
        with torch.no_grad():
            feats_vel = feats[..., 2:4]  
            if self.normalize and self.mean is not None and self.std is not None:
                B, T_obs, N, C = feats_vel.shape
                f2d = feats_vel.view(B * T_obs * N, C)
                f2d = (f2d - self.mean[2:]) / self.std[2:]
                feats_vel = f2d.view(B, T_obs, N, C)
            pred_vel_norm = self.model(feats_vel, adj, future_len)  
            if self.normalize and self.mean is not None and self.std is not None:
                B2, T2, N2, C2 = pred_vel_norm.shape
                p2d = pred_vel_norm.view(B2 * T2 * N2, C2)
                p2d = p2d * self.std[2:] + self.mean[2:]
                pred_vel = p2d.view(B2, T2, N2, C2)
            else:
                pred_vel = pred_vel_norm
        return pred_vel
