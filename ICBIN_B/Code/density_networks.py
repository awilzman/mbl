# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:45:17 2024

@author: Andrew
v3.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class tet10_encoder(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, num_exp=1, bidirectional=False):
        super(tet10_encoder, self).__init__()
        self.act = nn.LeakyReLU()
        self.lstm = nn.LSTM(
            input_size=31,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        # Output size depends on whether the LSTM is bidirectional
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_exp = num_exp
        
        self.experts_fc1 = nn.ModuleList([
            nn.Linear(fc_input_size, hidden_size) for _ in range(self.num_exp)])
        self.gating_network = nn.Linear(fc_input_size, self.num_exp)
        
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        
    def forward(self, x):
        # Calculate sequence lengths, assuming padding value is 0
        lengths = (x != 0).sum(dim=1)[:, 0].cpu()
    
        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
    
        # Unpack sequences to get the hidden states for all time steps
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    
        # Apply expert gating mechanism on the hidden states at each time step
        # (output is of shape [batch_size, seq_len, hidden_size])
        if self.lstm.bidirectional:
            # If bidirectional, concatenate the forward and backward hidden states at each time step
            output = torch.cat((output[:, :, :self.lstm.hidden_size], 
                                output[:, :, self.lstm.hidden_size:]), dim=2)
        
        gate_values = F.softmax(self.gating_network(output), dim=2)
    
        expert_outputs = [expert(output) for expert in self.experts_fc1]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        encoded = torch.einsum('bti,btij->btj', gate_values, expert_outputs)  
    
        encoded = self.fc2(self.act(encoded))
    
        return encoded
    
class tet10_decoder(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_decoder, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        self.act = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.codeword_size + 1, 64)
        self.fc2 = nn.Linear(65, 32)
        self.fc3 = nn.Linear(33, self.feature_size)
        
        
    def forward(self, x, i):
        i = i.unsqueeze(2)
        x = torch.cat((x, i), dim=2)  # Shape: [B, E, D + 1]
        
        x = self.act(self.fc1(x))
        x = torch.cat((x, i), dim=2)
        x = self.act(self.fc2(x))
        x = torch.cat((x, i), dim=2)
        x = self.fc3(x)
        x = torch.cat((x, i), dim=2)
        return x
    
class tet10_densify(nn.Module):
    def __init__(self, codeword_size=16, num_exp=1):
        super(tet10_densify, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 31
        
        self.num_exp = num_exp
        
        self.experts_fc1 = nn.ModuleList([nn.Linear(
            self.feature_size + self.codeword_size, codeword_size) for _ in range(self.num_exp)])
        self.gating_network = nn.Linear(self.feature_size + self.codeword_size, self.num_exp)
        
        self.fc2 = nn.Linear(1+self.codeword_size, self.codeword_size//2)
        self.fc3 = nn.Linear(1+self.codeword_size//2, 1)
        
    
    def forward(self, elems, encoded_features):
        # x shape: (B, E, 31) - Original features
        # encoded_features shape: (B, E, 16) - Encoded features
        B, E, _ = elems.size()
        
        x = torch.cat((elems, encoded_features), dim=2)  # (B, E, 31 + codeword)
        xs = elems[:,:,-1]
        gate_values = F.softmax(self.gating_network(x.view(-1, x.size(-1))), dim=1)  # Shape: [B*E, num_experts]
        gate_values = gate_values.view(B, E, self.num_exp)
        
        expert_outputs = [expert(x) for expert in self.experts_fc1]
        expert_outputs = torch.stack(expert_outputs, dim=2)  # Shape: [B, E, num_experts, codeword_size]

        # Weighted sum of expert outputs
        x = torch.einsum('bij,bijk->bik', gate_values, expert_outputs) # [B, E, codword size]
        x = torch.cat([x,xs.unsqueeze(2)],dim=2)
        x = torch.relu(self.fc2(x))
        x = torch.cat([x,xs.unsqueeze(2)],dim=2)
        x = torch.relu(self.fc3(x))
        return x