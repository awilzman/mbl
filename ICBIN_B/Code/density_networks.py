# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:45:17 2024

@author: Andrew
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class tet10_encoder(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, bidirectional=False):
        super(tet10_encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=30,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=0.2,
            batch_first=True
        )
        # Output size depends on whether the LSTM is bidirectional
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_exp = 2
        
        self.experts_fc1 = nn.ModuleList([
            nn.Linear(fc_input_size, hidden_size) for _ in range(self.num_exp)])
        self.gating_network = nn.Linear(fc_input_size, self.num_exp)
        
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        
    def forward(self, x):
        lengths = (x != 0).sum(dim=1)[:, 0].cpu()  # Keep lengths on CPU for packing

        # Pack padded sequences for variable-length inputs
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        
        # If LSTM is bidirectional, concatenate forward and backward hidden states
        if self.lstm.bidirectional:
            h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_n = h_n[-1, :, :]
            
        gate_values = F.softmax(self.gating_network(h_n), dim=1) 
        expert_outputs = [expert(h_n) for expert in self.experts_fc1]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        encoded = torch.einsum('bi,bij->bj', gate_values, expert_outputs)
        
        encoded = self.fc2(torch.relu(encoded))

        return encoded
    
class tet10_decoder(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_decoder, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        self.fc1 = nn.Linear(self.codeword_size + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.feature_size)
        
        
    def forward(self, x, i):
        x = x.unsqueeze(1)  # Shape: [B, 1, D]
        i = i.unsqueeze(-1)  # Shape: [B, E, 1]
        
        x = x.expand(-1, i.size(1), -1)  # Shape: [B, E, D]
        
        x = torch.cat((x, i), dim=-1)  # Shape: [B, E, D + 1]
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class tet10_densify(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_densify, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        self.num_exp = 2
        # Define the network layers
        self.experts_fc1 = nn.ModuleList([nn.Linear(
            self.feature_size + self.codeword_size, codeword_size) for _ in range(self.num_exp)])
        self.gating_network = nn.Linear(self.feature_size + self.codeword_size, self.num_exp)
        
        self.fc2 = nn.Linear(self.codeword_size, 1)
        
    def forward(self, x, encoded_features):
        # x shape: (B, E, 30) - Original features
        # encoded_features shape: (B, 1, 16) - Encoded features
        B, E, _ = x.size()
        
        encoded_features_repeated = encoded_features.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, E, 16)
        
        combined = torch.cat((x, encoded_features_repeated), dim=2)  # (B, E, 30 + 16)
        
        gate_values = F.softmax(self.gating_network(combined.view(-1, combined.size(-1))), dim=1)  # Shape: [B*E, num_experts]
        gate_values = gate_values.view(B, E, self.num_exp)
        
        expert_outputs = [expert(combined) for expert in self.experts_fc1]
        expert_outputs = torch.stack(expert_outputs, dim=2)  # Shape: [B, E, num_experts, codeword_size]

        # Weighted sum of expert outputs
        x = torch.einsum('bij,bijk->bik', gate_values, expert_outputs) 
        
        x = torch.relu(self.fc2(x))
        return x