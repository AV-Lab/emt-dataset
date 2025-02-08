#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:47:47 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim
from evaluation.distance_metrics import calculate_ade, calculate_fde
from tqdm import tqdm



class TransformerPredictor:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:x.size(0), :]

    class TransformerEncoder(nn.Module):
        def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
            super().__init__()
            self.embedding = nn.Linear(input_size, d_model)
            self.pos_encoder = TransformerPredictor.PositionalEncoding(d_model)
            encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
            self.dropout = nn.Dropout(dropout)
            self.d_model = d_model

        def forward(self, src, src_mask, src_padding_mask):
            src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            src = self.pos_encoder(src)
            src = self.dropout(src)
            output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_padding_mask)
            return output

    class TransformerDecoder(nn.Module):
        def __init__(self, output_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
            super().__init__()
            self.embedding = nn.Linear(output_size, d_model)
            self.pos_encoder = TransformerPredictor.PositionalEncoding(d_model)
            decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
            self.fc_out = nn.Linear(d_model, output_size)
            self.dropout = nn.Dropout(dropout)
            self.d_model = d_model

        def forward(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_padding_mask):
            tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            tgt = self.pos_encoder(tgt)
            tgt = self.dropout(tgt)
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=tgt_padding_mask,
                                              memory_key_padding_mask=memory_padding_mask)
            output = self.fc_out(output)
            return output

    class Seq2SeqTransformer(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
            memory = self.encoder(src, src_mask, src_padding_mask)
            output = self.decoder(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
            return output

    def __init__(self, model_parameters, loading_parameters, training_parameters, checkpoint_file=None):
        self.input_size = model_parameters['state_size']
        self.output_size = model_parameters['state_size']
        self.d_model = model_parameters['d_model']
        self.nhead = model_parameters['nhead']
        self.num_encoder_layers = model_parameters['num_encoder_layers']
        self.num_decoder_layers = model_parameters['num_decoder_layers']
        self.dim_feedforward = model_parameters['dim_feedforward']
        self.dropout = 0.1
        self.input_len = loading_parameters['observation_length']
        self.target_len = loading_parameters['prediction_horizon']
        self.num_epochs = training_parameters['num_epochs']
        self.patience = training_parameters['patience']
        self.device = training_parameters['device']
        self.best_val_loss = float('inf')

        encoder = self.TransformerEncoder(self.input_size, self.d_model, self.nhead, self.num_encoder_layers,
                                          self.dim_feedforward, self.dropout)
        decoder = self.TransformerDecoder(self.output_size, self.d_model, self.nhead, self.num_decoder_layers,
                                          self.dim_feedforward, self.dropout)
        self.model = self.Seq2SeqTransformer(encoder, decoder).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['learning_rate'])

        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def create_padding_mask(self, seq):
        return (seq == 0).transpose(0, 1)

    def save_checkpoint(self, saving_checkpoint_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, '{}/transformer_checkpoint.pth'.format(saving_checkpoint_path))
    
    def train(self, train_loader, valid_loader=None, saving_checkpoint_path=None):
        print("Total batches: ", len(train_loader))
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            for idx, batch in enumerate(train_loader):
                inputs, targets = batch
                
                # Transpose inputs and targets to match [seq_len, batch_size, feature_dim]
                inputs = inputs.transpose(0, 1).to(self.device)
                targets = targets.transpose(0, 1).to(self.device)
                
                # Create src and tgt padding masks based on the correct shape
                src_padding_mask = (inputs == 0).transpose(0, 1).any(dim=-1).to(self.device)  # [batch_size, seq_len]
                tgt_padding_mask = (targets == 0).transpose(0, 1).any(dim=-1).to(self.device)  # [batch_size, seq_len]
                
                tgt_padding_mask = tgt_padding_mask[:, :-1]
                
                # Zero grad
                self.optimizer.zero_grad()
                
                # Call the model
                outputs = self.model(
                    inputs,
                    targets[:-1],  # Shift targets
                    None,  # Optionally pass masks
                    None,
                    src_padding_mask,  # Correct padding mask shape
                    tgt_padding_mask   # Correct padding mask shape
                )
                
                # Compute loss
                loss = self.criterion(outputs, targets[1:])
                loss.backward()
                
                # Step optimizer
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
        self.model.eval()
        epoch_loss = 0
        num_batches = len(valid_loader)
    
        with torch.no_grad():
            for inputs, targets in valid_loader:
                # Transpose inputs and targets
                inputs = inputs.transpose(0, 1).to(self.device)
                targets = targets.transpose(0, 1).to(self.device)
    
                # Prepare padding masks and ensure they are 2D
                src_padding_mask = (inputs == 0).transpose(0, 1).any(dim=-1).to(self.device)  # [batch_size, seq_len]
                tgt_padding_mask = (targets == 0).transpose(0, 1).any(dim=-1).to(self.device)  # [batch_size, seq_len]
                tgt_padding_mask = tgt_padding_mask[:, :-1]
                outputs = self.model(inputs, targets[:-1], None, None, src_padding_mask, tgt_padding_mask)
    
                # Compute the loss
                loss = self.criterion(outputs, targets[1:])
                epoch_loss += loss.item()
    
        avg_epoch_loss = epoch_loss / num_batches
        
        print(f'Validation Loss: {avg_epoch_loss:.4f}')
        return avg_epoch_loss


    def predict(self, input_trajectory, target_len=None):
        if target_len is None:
            target_len = self.target_len
        
        self.model.eval()
    
        # Start with an empty target tensor (batch_size, 1, output_size)
        batch_size = input_trajectory.size(0)
        tgt = torch.zeros(batch_size, 1, self.output_size).to(self.device)
    
        # Use boolean masks for padding, where True indicates padding
        src_padding_mask = (input_trajectory.abs().sum(dim=-1) == 0)  # Shape: (batch_size, input_len)
        tgt_padding_mask = torch.zeros(batch_size, 1).bool().to(self.device)  # Initially no padding for tgt
    
        with torch.no_grad():
            for _ in range(target_len):
                # Predict the next timestep
                output = self.model(
                    input_trajectory.transpose(0, 1).to(self.device),  # src
                    tgt.transpose(0, 1).to(self.device),               # tgt
                    None,                                              # src_mask
                    None,                                              # tgt_mask
                    src_padding_mask.to(self.device),                  # src_padding_mask
                    tgt_padding_mask.to(self.device)                   # tgt_padding_mask
                )
                
                # Extract the last predicted timestep from the output
                last_step = output[-1:, :, :]  # Shape: (1, batch_size, output_size)
    
                # Append the last predicted step to the target sequence
                tgt = torch.cat((tgt, last_step.transpose(0, 1)), dim=1)  # Concatenate along the sequence length axis
    
                # Update tgt_padding_mask by adding zeros for the new sequence
                tgt_padding_mask = torch.cat(
                    (tgt_padding_mask, torch.zeros(batch_size, 1).bool().to(self.device)), dim=1
                )
    
            return tgt[:, 1:, :]  # Return the sequence without the initial zeroed target




    def evaluate(self, loader):
        self.model.eval()
        ade = 0
        fde = 0
    
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                
                # Transpose inputs to match model expectations (sequence_len, batch_size, features)
                inputs = inputs.transpose(0, 1).to(self.device)
                targets = targets.transpose(0, 1).to(self.device)
                
                # Predict future trajectory based on the input trajectory
                predicted_trajectory = self.predict(inputs.transpose(0, 1), target_len=targets.size(0))
                
                # Predicted shape: [batch_size, seq_len, features] --> Transpose to [seq_len, batch_size, features]
                predicted_trajectory = predicted_trajectory.transpose(0, 1)  # Now: [seq_len, batch_size, features]
                
                # Compute ADE and FDE
                ade += calculate_ade(predicted_trajectory, targets.to(self.device))
                fde += calculate_fde(predicted_trajectory, targets.to(self.device))
        
        # Normalize by the number of batches
        ade /= len(loader)
        fde /= len(loader)
    
        print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")
        return ade, fde

