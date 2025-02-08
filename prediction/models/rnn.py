#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:52 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim
from evaluation.distance_metrics import calculate_ade, calculate_fde
import torch.distributions as dist


class RNNPredictor:    
    class MultiEncoder(nn.Module):    
        def __init__(self, input_size, hidden_size, num_layers):
            """+           Initializes the LSTM encoder.
            
            Args:
                input_size (int): Number of input features.
                hidden_size (int): Number of hidden units in LSTM.
                num_layers (int): Number of LSTM layers.
            """
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        def forward(self, x):
            """
            Forward pass for the encoder.
            
            Args:
                x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            
            Returns:
                hidden (Tensor): Hidden states from the LSTM.
                cell (Tensor): Cell states from the LSTM.
            """
            outputs, (hidden, cell) = self.lstm(x)
            return hidden, cell
    
    class MultiDecoder(nn.Module):    
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            """
            Initializes the LSTM decoder.
            
            Args:
                input_size (int): Number of input features.
                hidden_size (int): Number of hidden units in LSTM.
                output_size (int): Number of output features.
                num_layers (int): Number of LSTM layers.
            """
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden, cell):
            """
            Forward pass for the decoder.
            
            Args:
                x (Tensor): Input tensor of shape (batch_size, 1, input_size).
                hidden (Tensor): Hidden states from the LSTM encoder.
                cell (Tensor): Cell states from the LSTM encoder.
            
            Returns:
                predictions (Tensor): Output predictions of shape (batch_size, 1, output_size).
                hidden (Tensor): Updated hidden states.
                cell (Tensor): Updated cell states.
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
                source (Tensor): Input sequence tensor of shape (batch_size, seq_len, input_size).
                target_len (int): Length of the target sequence.
            
            Returns:
                outputs (Tensor): Predicted sequence tensor of shape (batch_size, target_len, output_size).
            """
            batch_size = source.size(0)
            hidden, cell = self.encoder(source)
            
            decoder_input = torch.zeros(batch_size, 1, source.size(2)).to(source.device)
            outputs = []
            
            for _ in range(target_len):
                decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                outputs.append(decoder_output)
                decoder_input = decoder_output
            
            outputs = torch.cat(outputs, dim=1)
            return outputs

    def __init__(self, observation_length, prediction_horizon, checkpoint_file=None):
        """
        Initializes the RNNPredictor class with encoder, decoder, and training configurations.
        
        Args:
            model_parameters (dict): Contains model configuration (e.g., hidden_size, num_layers, etc.).
            loading_parameters (dict): Contains data loading configuration (e.g., observation_length, prediction_horizon).
            training_parameters (dict): Contains training configuration (e.g., num_epochs, learning_rate).
            checkpoint_file (str, optional): Path to the model checkpoint file.
        """
        self.hidden_size = 128
        self.num_layers = 2
                
        self.input_size = 2
        self.output_size = 2
        self.input_len = observation_length
        self.target_len = prediction_horizon

        self.num_epochs = 300
        self.learning_rate = 0.001
        self.patience = 5
        self.device = 0
        self.best_val_loss = float('inf')
        
        encoder = self.MultiEncoder(self.input_size, self.hidden_size, self.num_layers)
        decoder = self.MultiDecoder(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.model = self.Seq2Seq(encoder, decoder).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if checkpoint_file is not None:
            print('Loading weights from checkpoint')
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_checkpoint(self, saving_checkpoint_path):
        """
        Saves the current model and optimizer states to a checkpoint file.
        
        Args:
            saving_checkpoint_path (str): Path to save the checkpoint.
        """
        print("Saving the checkpoint ....")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{saving_checkpoint_path}/trained_model.pth')
    
    def train(self, train_loader, valid_loader=None, saving_checkpoint_path=None):
        """
        Trains the model using the given training data loader and optionally validates using a validation loader.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader, optional): DataLoader for validation data.
            saving_checkpoint_path (str, optional): Path to save the best checkpoint.
        """
        print("Total batches: ", len(train_loader))
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            for idx, batch in enumerate(train_loader):
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs.to(self.device), self.target_len)
                loss = self.criterion(outputs, targets.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_epoch_loss:.4f}')
            
            if valid_loader is not None:
                val_loss = self.validate(valid_loader)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered.")
                    if saving_checkpoint_path is not None:
                        self.save_checkpoint(saving_checkpoint_path)
                    return
                
        if saving_checkpoint_path is not None:
            self.save_checkpoint(saving_checkpoint_path)
        
    def validate(self, valid_loader):
        """
        Validates the model using the validation data loader.
        
        Args:
            valid_loader (DataLoader): DataLoader for validation data.
        
        Returns:
            float: Average validation loss over all batches.
        """
        self.model.eval()
        epoch_loss = 0
        num_batches = len(valid_loader)

        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = self.model(inputs.to(self.device), self.target_len)
                loss = self.criterion(outputs, targets.to(self.device))
                epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f'Validation Loss: {avg_epoch_loss:.4f}')
        return avg_epoch_loss

    def evaluate(self, loader):
        """
        Evaluates the model on a test dataset by calculating Average Displacement Error (ADE) and Final Displacement Error (FDE).
        
        Args:
            loader (DataLoader): DataLoader for the test data.
        
        Returns:
            float: ADE (Average Displacement Error).
            float: FDE (Final Displacement Error).
        """
        ade = 0
        fde = 0
        print("Total batches: ", len(loader))
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                outputs = self.predict(inputs.to(self.device), self.target_len)
                ade += calculate_ade(outputs, targets.to(self.device))
                fde += calculate_fde(outputs, targets.to(self.device))
        
        ade = ade/len(loader)
        fde = fde/len(loader)
        
        print("ADE: {:.4f}, FDE: {:.4f}".format(ade, fde))
        return ade, fde
        
    def predict(self, input_trajectory, target_len=10):
        """
        Predicts the future trajectory given the input trajectory.
        
        Args:
            input_trajectory (Tensor): Input sequence of shape (batch_size, seq_len, input_size).
            target_len (int, optional): Number of future timesteps to predict. Defaults to 10.
        
        Returns:
            Tensor: Predicted trajectory of shape (batch_size, target_len, output_size).
        """
        
        if input_trajectory.shape[1] < self.input_len:
            zero_tensor = torch.tensor([[0.0, 0.0]])
            num_to_add = self.input_len - input_trajectory.shape[1]
            padding = zero_tensor.repeat(num_to_add, 1)
            input_trajectory = torch.cat((padding, input_trajectory), dim=0)
            
        self.model.eval()
        with torch.no_grad():
            return self.model(input_trajectory.to(self.device), target_len)
        
        


