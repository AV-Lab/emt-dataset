import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.classification_metrics import compute_precision_recall_f1
from evaluation.distance_metrics import compute_intention_and_distance_metrics
from constants import LABEL_TO_CLASS

class RNNVanillaPredictor:
    """
    An LSTM-based model for multi-step intention classification, 
    optionally normalizing the input features (x,y).
    """

    class MultiEncoder(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        def forward(self, x):
            _, (hidden, cell) = self.lstm(x)
            return hidden, cell

    class MultiDecoder(nn.Module):
        def __init__(self, hidden_size, output_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, hidden, cell, target_len):
            """
            Decode for 'target_len' steps, each step returning class logits of shape (batch,1,out_size).
            Uses a zero input for each step.
            """
            batch_size = hidden.shape[1]
            input_dim = hidden.shape[2]
            decoder_input = torch.zeros(batch_size, 1, input_dim, device=hidden.device)
            
            outputs = []
            for _ in range(target_len):
                dec_out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
                logits = self.fc(dec_out)  # => (batch_size,1,output_size)
                outputs.append(logits)
            return torch.cat(outputs, dim=1)  # => (batch_size, target_len, output_size)

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, target_len):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.target_len = target_len

        def forward(self, x):
            hidden, cell = self.encoder(x)
            out = self.decoder(hidden, cell, self.target_len)
            return out

    def __init__(self, observation_length, prediction_horizon, device, normalize=False, checkpoint_file=None):
        """
        Args:
            observation_length  (int): # of input frames
            prediction_horizon  (int): # of future frames to predict
            num_classes         (int): # of distinct intention classes
            normalize           (bool): if True, do (x-mean)/std on input
            checkpoint_file     (str): optional path to saved checkpoint
        """
        self.input_size = 2
        self.hidden_size = 128
        self.num_layers  = 2
        self.num_classes = 11
        self.observation_len = observation_length
        self.target_len  = prediction_horizon
        self.normalize   = normalize

        self.num_epochs  = 100
        self.device = device

        # Build the model
        enc = self.MultiEncoder(self.input_size, self.hidden_size, self.num_layers)
        dec = self.MultiDecoder(self.hidden_size, self.num_classes, self.num_layers)
        self.model = self.Seq2Seq(enc, dec, self.target_len).to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        if checkpoint_file:
            cpt = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(cpt["model_state_dict"])
            self.optimizer.load_state_dict(cpt["optimizer_state_dict"])
            if self.normalize:
                self.mean = cpt["mean"]
                self.std  = cpt["std"]


    def train(self, train_loader, valid_loader=None):
        """
        Train loop:
          - If self.normalize==True, do (inputs - mean)/std before forward
        """
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device)
            self.std  = train_loader.dataset.std.to(self.device)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            load_train = tqdm(train_loader, desc=f"Epoch: {epoch+1}/{self.num_epochs}", leave=False)
    
            for id_b, batch in enumerate(load_train):
                obs_tensor, targets = batch
                obs_tensor = obs_tensor.to(self.device)  # shape [B, obs_seq_len, 4]
                targets = targets.to(self.device)  # shape [B, pred_seq_len, 4]
                
                if self.normalize:
                    input_vel = obs_tensor[:, 1:, 2:4]  # skip first frame’s velocity
                    input_vel_norm = (input_vel - self.mean[2:]) / self.std[2:]
                else:
                    input_vel_norm = obs_tensor[:, 1:, 2:4]

                self.optimizer.zero_grad()

                outputs = self.model(input_vel_norm)  # shape (batch_size, target_len, num_classes)

                loss = self.criterion(outputs.view(-1, self.num_classes), targets.argmax(dim=-1).view(-1))

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            if valid_loader is not None:
                val_loss = self.evaluate(valid_loader)
                print(f"Validation Loss: {val_loss:.4f}")
                
        self.save_checkpoint()

    def evaluate(self, loader):
        """
        Evaluates the model and computes accuracy, precision, recall, and F1-score.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for obs_tensor, targets in loader:
                obs_tensor, targets = obs_tensor.to(self.device), targets.to(self.device)
                
                if self.normalize:
                    # Input velocities and target velocities
                    input_vel = obs_tensor[:, 1:, 2:4]  # skip first frame’s velocity
                    input_vel_norm = (input_vel - self.mean[2:]) / self.std[2:]
                else:
                    input_vel_norm = obs_tensor[:, 1:, 2:4]

                outputs = self.model(input_vel_norm)  # (batch_size, target_len, num_classes)
                loss = self.criterion(outputs.view(-1, self.num_classes), targets.argmax(dim=-1).view(-1))
                total_loss += loss.item()

                preds = outputs.argmax(dim=2)  # Convert logits to predicted class indices
                correct += (preds == targets.argmax(dim=2)).sum().item()
                total_samples += targets.numel() // self.num_classes  # Adjust for multiple timesteps
                
                all_preds.append(preds)
                all_targets.append(targets)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = compute_intention_and_distance_metrics(all_preds, all_targets, self.num_classes)
        
        print("F1 (All Timestamps) Overall (Macro):", metrics["f1_all_overall"])
        print("F1 (All Timestamps) Per Class:")
        for c in range(self.num_classes):
            print(f"  [Class {c}]: {metrics['f1_all_per_class'][c]:.4f}")
        
        print("F1 (Last Token) Overall (Macro):", metrics["f1_last_overall"])
        print("F1 (Last Token) Per Class:")
        for c in range(self.num_classes):
            print(f"  [Class {c}]: {metrics['f1_last_per_class'][c]:.4f}")
        
        print("Average Normalized Levenshtein Distance:", metrics["avg_norm_lev_distance"])
        
        avg_loss = total_loss / len(loader)
        print(f"Test Loss: {avg_loss:.4f}")
        
        return metrics


    def save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            checkpoint["mean"] = self.mean
            checkpoint["std"]  = self.std

        torch.save(checkpoint, "rnn_intention_model.pth")
        print("Model checkpoint saved")


    
    def predict(self, input_trajectories):
        """
        Predicts future intentions for multiple objects given their absolute trajectories.
    
        Args:
            input_trajectories (list of lists of tuples): Observed absolute trajectories for multiple objects.
            Each object has a trajectory [(x1, y1), (x2, y2), ...], and trajectories may have different lengths.
    
        Returns:
            Tensor: Predicted class indices of shape (num_objects, target_len).
        """
        self.model.eval()
        processed_velocities = []
    
        for trajectory in input_trajectories:
            traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=self.device)
    
            if traj_tensor.shape[0] == 1:
                processed_velocities.append(torch.zeros((self.observation_len - 1, 2), dtype=torch.float32, device=self.device))
                continue
    
            obs_vel = traj_tensor[1:] - traj_tensor[:-1]
            if len(obs_vel) < 0:
                obs_vel = torch.zeros_like(traj_tensor)
                              
            if obs_vel.shape[0] < self.observation_len - 1:
                pad_size = self.observation_len - 1 - obs_vel.shape[0]
                pad = torch.zeros((pad_size, 2), dtype=torch.float32, device=self.device)
                obs_vel = torch.cat([pad, obs_vel], dim=0)

            processed_velocities.append(obs_vel)

        velocities_batch = torch.stack(processed_velocities, dim=0)
    
        with torch.no_grad():
            outputs = self.model(velocities_batch)
            preds = outputs.argmax(dim=2)
        preds = [[LABEL_TO_CLASS[p.item()] for p in pred] for pred in preds]
    
        return preds  


        
