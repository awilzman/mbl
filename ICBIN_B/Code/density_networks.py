# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:45:17 2024

@author: Andrew
"""
import torch
import torch.nn as nn

class tet10_encoder(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, bidirectional=False):
        super(tet10_encoder, self).__init__()
        self.gru = nn.GRU(
            input_size=30,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.1)
        # Output size depends on whether the GRU is bidirectional
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, lengths):
        # Pack padded sequences for variable-length inputs
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # Unpack the sequence

        last_hidden_states = output[torch.arange(output.size(0)), lengths - 1, :]
        encoded = self.fc1(last_hidden_states)
        
        encoded = self.fc2(torch.relu(self.dropout(encoded)))
        encoded = self.fc3(torch.relu(self.dropout(encoded)))

        return encoded
    
class tet10_decoder(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_decoder, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        self.fc1 = nn.Linear(self.codeword_size + 1, 16)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(16, self.feature_size)
        
    def forward(self, x, i):
        x = x.unsqueeze(1)  # Shape: [B, 1, D]
        i = i.unsqueeze(-1)  # Shape: [B, E, 1]
        
        x = x.expand(-1, i.size(1), -1)  # Shape: [B, E, D]
        
        x = torch.cat((x, i), dim=-1)  # Shape: [B, E, D + 1]
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
    
class tet10_densify(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_densify, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        # Define the network layers
        self.fc1 = nn.Linear(self.feature_size + self.codeword_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x, encoded_features):
        # x shape: (B, E, 30) - Original features
        # encoded_features shape: (B, 1, 16) - Encoded features
        B, E, _ = x.size()
        
        encoded_features_repeated = encoded_features.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, E, 16)
        
        combined = torch.cat((x, encoded_features_repeated), dim=2)  # (B, E, 30 + 16)
        
        x = self.fc1(combined)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        
        return x