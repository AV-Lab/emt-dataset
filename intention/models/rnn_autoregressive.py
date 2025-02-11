import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.classification_metrics import compute_precision_recall_f1


class RNNAutoregressivePredictor:
    class MultiEncoder(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

        def forward(self, x):
            """
            x: (batch_size, obs_len, input_size).
            We only need the final hidden and cell states
            """
            _, (hidden, cell) = self.lstm(x)
            return hidden, cell


    class MultiDecoder(nn.Module):
        def __init__(self, hidden_size, output_size, num_layers):
            super().__init__()
            self.class_embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, hidden, cell, target_len, target_seq=None, teacher_forcing_ratio=0.0):
            """
            hidden, cell: (num_layers, batch_size, hidden_size)
            target_seq: (batch_size, target_len) integer class labels
            """
            batch_size = hidden.shape[1]

            outputs = []
            decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=hidden.device)

            for t in range(target_len):
                embedded_input = self.class_embedding(decoder_input)
                lstm_out, (hidden, cell) = self.lstm(embedded_input, (hidden, cell))
                step_logits = self.fc(lstm_out)
                outputs.append(step_logits)
                preds = step_logits.argmax(dim=2)  # (batch_size, 1)                
                if (target_seq is not None) and (random.random() < teacher_forcing_ratio):
                    next_input = target_seq[:, t].unsqueeze(1)  # shape => (batch_size, 1)
                else:
                    next_input = preds                    
                decoder_input = next_input
            outputs = torch.cat(outputs, dim=1)
            return outputs

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, target_len):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.target_len = target_len
            
        def forward(self, source, target_seq=None, teacher_forcing_ratio=0.0):
            hidden, cell = self.encoder(source)
            outputs = self.decoder(hidden, cell, self.target_len, target_seq, teacher_forcing_ratio)
            return outputs

    def __init__(self, observation_length, prediction_horizon, device, normalize=False, checkpoint_file=None):
        self.input_size = 2  # (x, y) past trajectory
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = 11
        self.num_epochs = 100
        self.normalize = normalize
        self.observation_len = observation_length
        self.target_len = prediction_horizon
        self.device = device

        encoder = self.MultiEncoder(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        decoder = self.MultiDecoder(hidden_size=self.hidden_size, output_size=self.num_classes, num_layers=self.num_layers)
        self.model = self.Seq2Seq(encoder, decoder, self.target_len).to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        if checkpoint_file:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.normalize:
                self.mean = checkpoint["mean"]
                self.std  = checkpoint["std"]

    def train(self, train_loader):
        self.model.train()
        
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device)
            self.std  = train_loader.dataset.std.to(self.device)
            
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).long()
                
                if self.normalize:
                    input_vel = inputs[:, 1:, 2:4]  # skip first frame’s velocity
                    input_vel_norm = (input_vel - self.mean[2:]) / self.std[2:]
                else:
                    input_vel_norm = inputs[:, 1:, 2:4]
                
                self.optimizer.zero_grad()
                outputs = self.model(input_vel_norm, target_seq=targets, teacher_forcing_ratio=0.0)
                loss = self.criterion(outputs.view(-1, self.num_classes), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
            
        self.save_checkpoint()

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.float().to(self.device)
                targets = targets.to(self.device).long()
                
                if self.normalize:
                    input_vel = inputs[:, 1:, 2:4]  # skip first frame’s velocity
                    input_vel_norm = (input_vel - self.mean[2:]) / self.std[2:]
                else:
                    input_vel_norm = inputs[:, 1:, 2:4]

                # teacher_forcing_ratio=0 => pure autoregressive
                outputs = self.model(input_vel_norm, target_seq=None, teacher_forcing_ratio=0.0)
                loss = self.criterion(outputs.view(-1, self.num_classes), targets.view(-1))
                total_loss += loss.item()

                preds = outputs.argmax(dim=2)
                correct += (preds == targets).sum().item()
                total_samples += targets.numel()

                all_preds.append(preds)
                all_targets.append(targets)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # After gathering all_preds and all_targets
        (precision_overall, 
         recall_overall, 
         f1_overall, 
         precision_per_class, 
         recall_per_class, 
         f1_per_class) = compute_precision_recall_f1(all_preds, all_targets, self.num_classes)
    
        # Then you can log/print them as you wish:
        print("Overall Precision: ", precision_overall)
        print("Overall Recall   : ", recall_overall)
        print("Overall F1       : ", f1_overall)
    
        for c in range(self.num_classes):
            print(f"[Class {c}] "
                  f"Precision: {precision_per_class[c]:.4f}, "
                  f"Recall: {recall_per_class[c]:.4f}, "
                  f"F1: {f1_per_class[c]:.4f}")
        
        avg_loss = total_loss / len(loader)
        print(f"Test Loss: {avg_loss:.4f}")
        
        return (precision_overall, recall_overall, f1_overall, precision_per_class, recall_per_class, f1_per_class) 
    
    def save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            checkpoint["mean"] = self.mean
            checkpoint["std"]  = self.std

        torch.save(checkpoint, "rnn_intention_model_autoregeressive.pth")
        print("Model checkpoint saved")

    def predict(self, input_trajectory):
        """
        Predicts the future intentions given an absolute trajectory.
    
        Args:
            input_trajectory (list of tuples): Observed absolute trajectory [(x1, y1), (x2, y2), ...]
    
        Returns:
            Tensor: Predicted class indices of shape (decode_len,)
        """
        self.model.eval()
        
        input_trajectory = torch.tensor(input_trajectory, dtype=torch.float32, device=self.device)
        if input_trajectory.shape[0] == 1:
            return torch.zeros(self.target_len, dtype=torch.long, device=self.device)
        velocities = input_trajectory[1:] - input_trajectory[:-1]
    
        # Pad from the beginning if needed
        if velocities.shape[0] < self.observation_len - 1:
            pad_size = self.observation_len - 1 - velocities.shape[0]
            pad = torch.zeros((pad_size, 2), dtype=torch.float32, device=self.device)
            velocities = torch.cat([pad, velocities], dim=0)  # Pad before the sequence
            
        if self.normalize and self.mean is not None and self.std is not None:
            velocities = (velocities - self.mean[2:]) / self.std[2:]    
    
        velocities = velocities.unsqueeze(0)  # Add batch dimension
    
        with torch.no_grad():
            outputs = self.model(velocities, target_seq=None, teacher_forcing_ratio=0.0)
            preds = outputs.argmax(dim=2)  # Get predicted class indices
    
        return preds.squeeze(0)  # Remove batch dimension, shape: (decode_len,)

