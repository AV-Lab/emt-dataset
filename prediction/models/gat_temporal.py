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

class GATLayer(nn.Module):
    """
    A Graph Attention Network (GAT) layer.

    This layer applies a linear transformation to the input features and computes attention
    coefficients between nodes, aggregating neighbor information accordingly.

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per head.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability on the attention coefficients.
        concat (bool, optional): If True, concatenate the outputs of the heads; if False, average them.

    Returns:
        torch.Tensor: Output features of shape (B, N, num_heads*out_features) if concat=True,
                      or (B, N, out_features) if concat=False.
    """
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.0, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, num_heads))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Input features of shape (B, N, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (B, N, N) (assumed binary).
        
        Returns:
            torch.Tensor: Output features.
        """
        B, N, _ = x.size()
        h = self.W(x) 
        h = h.view(B, N, self.num_heads, self.out_features) 
        h_i = h.unsqueeze(2).expand(B, N, N, self.num_heads, self.out_features)
        h_j = h.unsqueeze(1).expand(B, N, N, self.num_heads, self.out_features)
        a_input = torch.cat([h_i, h_j], dim=-1) 
        e = self.leakyrelu(torch.einsum("bijnk,kn->bijn", a_input, self.a)) 
        e_max, _ = e.max(dim=2, keepdim=True)
        e = e - e_max
        e = torch.clamp(e, min=-1e9)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(-1) > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=2)
        
        if self.dropout:
            attention = nn.functional.dropout(attention, p=self.dropout, training=self.training)

        h_prime = torch.einsum("bijn,bjnd->bind", attention, h)
        if self.concat:
            h_prime = h_prime.reshape(B, N, self.num_heads * self.out_features)
        else:
            h_prime = h_prime.mean(dim=1)
        return h_prime

class GATEncoder(nn.Module):
    """
    GAT-based encoder applied per time step.
    
    Args:
        in_features (int): Number of input features per node.
        hidden_size (int): Desired output dimension per head.
        num_layers (int, optional): Number of GAT layers.
        num_heads (int, optional): Number of attention heads per layer.
        dropout (float, optional): Dropout probability.
    
    Returns:
        torch.Tensor: Encoded features of shape (B, N, num_heads*hidden_size).
    """
    def __init__(self, in_features, hidden_size, num_layers=2, num_heads=1, dropout=0.0):
        super().__init__()
        layers = []
        layers.append(GATLayer(in_features, hidden_size, num_heads=num_heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 1):
            layers.append(GATLayer(hidden_size * num_heads, hidden_size, num_heads=num_heads, dropout=dropout, concat=True))
        self.gat_layers = nn.ModuleList(layers)
        self.activation = nn.ELU()
    
    def forward(self, x, adj):
        for layer in self.gat_layers:
            x = layer(x, adj)
            x = self.activation(x)
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM-based temporal encoder.
    
    Processes a sequence of per-node features and outputs a final hidden state.
    
    Args:
        input_size (int): Dimension of per-time-step features (from GAT).
        hidden_size (int): Hidden state dimension.
        num_layers (int, optional): Number of LSTM layers. Default is 2.
    
    Returns:
        torch.Tensor: Final hidden state of shape (B*N, hidden_size).
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return h_n[-1]

class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder.
    
    Autoregressively predicts future velocities for each node.
    
    Args:
        hidden_size (int): Hidden state dimension (should match encoder output).
        out_size (int): Output dimension (e.g., 2 for [vx, vy]).
        num_layers (int, optional): Number of LSTM layers. Default is 2.
    
    Returns:
        torch.Tensor: Predicted sequence of shape (B, future_len, N, out_size).
    """
    def __init__(self, hidden_size, out_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=out_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        self.start_token = nn.Parameter(torch.zeros(1, out_size))
    
    def forward(self, h0, future_len, B, N):
        h0 = h0.unsqueeze(0).expand(self.num_layers, B * N, self.hidden_size).contiguous()
        c0 = torch.zeros_like(h0)
        start = self.start_token.expand(B * N, -1).unsqueeze(1)  
        outputs = []
        input_lstm = start
        hidden = (h0, c0)
        for t in range(future_len):
            out, hidden = self.lstm(input_lstm, hidden)
            pred = self.fc(out)
            outputs.append(pred)
            input_lstm = pred
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(B, N, future_len, self.out_size).permute(0, 2, 1, 3).contiguous()
        return outputs


class GATLSTMSeq2Seq(nn.Module):
    """
    A GAT-LSTM Seq2Seq model.
    
    This model first applies a GAT encoder on each observed frame to extract spatial features.
    Then, an LSTM encoder aggregates the temporal sequence per node.
    Finally, an LSTM decoder autoregressively predicts future velocities.
    
    Args:
        gat_encoder (nn.Module): The GAT encoder module.
        lstm_encoder (nn.Module): The LSTM encoder module.
        lstm_decoder (nn.Module): The LSTM decoder module.
    
    Returns:
        torch.Tensor: Predicted sequence of shape (B, future_len, N, out_size).
    """
    def __init__(self, gat_encoder, lstm_encoder, lstm_decoder):
        super().__init__()
        self.gat_encoder = gat_encoder
        self.lstm_encoder = lstm_encoder
        self.lstm_decoder = lstm_decoder
    
    def forward(self, feats, adj, future_len):
        B, T_obs, N, in_size = feats.shape
        feats_reshaped = feats.view(B * T_obs, N, in_size)
        adj_expanded = adj.unsqueeze(1).expand(B, T_obs, N, N).contiguous().view(B * T_obs, N, N)
        gat_out = self.gat_encoder(feats_reshaped, adj_expanded)
        gat_hidden = gat_out.shape[-1]
        gat_seq = gat_out.view(B, T_obs, N, gat_hidden)
        lstm_in = gat_seq.transpose(1, 2).contiguous().view(B * N, T_obs, gat_hidden)
        h_enc = self.lstm_encoder(lstm_in) 
        outputs = self.lstm_decoder(h_enc, future_len, B, N)
        return outputs


class GATLSTMPredictor:
    """
    GAT-based Seq2Seq predictor that trains only on velocity from [x,y,vx,vy].
    
    Data assumptions:
      - feats:   (B, T_obs, N, 4)  -> [x, y, vx, vy]
      - adj:     (B, N, N)
      - targets: (B, T_future, N, 4) -> [x, y, vx, vy] for future steps
      - mask:    (B, N)
    
    The model uses only the velocity channels (channels 2:4) as input and computes a masked MSE loss
    on velocities. For evaluation (ADE/FDE), the predicted velocities are integrated into absolute positions
    using the last observed positions (channels 0:2 of feats). Normalization (if enabled) is applied only on
    the velocity channels.
    """
    def __init__(self, observation_length, future_length, max_nodes, device, normalize, checkpoint_file=None):
        """
        Args:
          observation_length (int): Number of observed time steps (T_obs)
          future_length (int): Number of future velocity steps (T_future)
          max_nodes (int): Maximum number of agents
          device (str or torch.device)
          normalize (bool): If True, apply normalization to (vx, vy)
          checkpoint_file (str): Optional checkpoint to load.
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

        self.gat_hidden = 256
        self.num_gat_layers = 2
        self.lstm_encode_num_layers = 2
        self.lstm_decode_num_layers = 2
        self.num_heads = 2
        self.lstm_hidden = 128
        self.in_size = 2
        self.out_size = 2
        self.lstm_encoder_input = self.gat_hidden * self.num_heads

        gat_encoder = GATEncoder(self.in_size, self.gat_hidden, num_layers=self.num_gat_layers, num_heads=self.num_heads, dropout=0.1)
        lstm_encoder = LSTMEncoder(input_size=self.lstm_encoder_input, hidden_size=self.lstm_hidden, num_layers=self.lstm_encode_num_layers)
        lstm_decoder = LSTMDecoder(hidden_size=self.lstm_hidden, out_size=self.out_size, num_layers=self.lstm_decode_num_layers)
        self.model = GATLSTMSeq2Seq(gat_encoder, lstm_encoder, lstm_decoder).to(self.device)

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
            ckpt["std"] = self.std
        torch.save(ckpt, f"{folder}/trained_gat_lstm_model.pth")
        print(f"Checkpoint saved to {folder}/trained_gat_lstm_model.pth")

    def train(self, train_loader, valid_loader=None, saving_checkpoint_path=None):
        """
        Expects each batch to yield:
          feats:   (B, T_obs, N, 4) – if T_obs is missing, add one.
          adj:     (B, N, N)
          targets: (B, T_future, N, 4) – we use only channels 2:4.
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
                    'ADE': f"{ade_batch:.4f}",
                    'FDE': f"{fde_batch:.4f}"
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
            avg_epoch_ade = epoch_ade / len(train_loader)
            avg_epoch_fde = epoch_fde / len(train_loader)
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