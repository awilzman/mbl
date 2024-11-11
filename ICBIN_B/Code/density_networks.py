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
    def __init__(self, hidden_size=16, num_layers=1, bidirectional=False):
        super(tet10_encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=31,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        # Output size depends on whether the LSTM is bidirectional
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.conv = nn.Conv1d(fc_input_size,hidden_size,1)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size,hidden_size)
        
    def forward(self, x):
        # Calculate sequence lengths, assuming padding value is 0
        lengths = (x != 0).sum(dim=1)[:, 0].cpu()
    
        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
    
        # Unpack sequences to get the hidden states for all time steps
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    
        # (output is of shape [batch_size, seq_len, hidden_size])
        if self.lstm.bidirectional:
            # If bidirectional, concatenate the forward and backward hidden states at each time step
            x = torch.cat((x[:, :, :self.lstm.hidden_size],
                           x[:, :, self.lstm.hidden_size:]), dim=2)
        x=x.permute(0,2,1)
        x = self.bn(self.conv(x))
        x=x.permute(0,2,1)
        x = F.relu(self.fc(x))
        
        return x, lengths
    
class tet10_decoder(nn.Module):
    def __init__(self, hidden_size=16, max_points=1024):
        super(tet10_decoder, self).__init__()
        
        self.decode_codeword = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, max_points)
        )
        
        self.decode_points = nn.Sequential(
            nn.Linear(1, 4),
            nn.SELU(),
            nn.Linear(4, 8),
            nn.SELU(),
            nn.Linear(8, 16),
            nn.SELU(),
            nn.Linear(16, 31)
        )
        
        self.sig = nn.Sigmoid()

    def forward(self, x, elems):
        x = self.decode_codeword(x).unsqueeze(-1)        
        x = self.decode_points(x)
        cortical = self.sig(x[:, :, -1])
        x = torch.cat((x[:, :, :-1], cortical.unsqueeze(-1)), dim=-1)

        # Resample to [batch_size, elems, 31]
        x = F.interpolate(x, size=(elems, 31), mode='linear', align_corners=False)
        return x
    
class tet10_densify(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_densify, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        self.act = nn.LeakyReLU()

        # Cortical
        self.cort_mlp1 = nn.Linear(self.feature_size + codeword_size, self.codeword_size,1)
        self.cort_conv2 = nn.Conv1d(self.codeword_size, self.codeword_size//4,1)
        self.cort_conv3 = nn.Conv1d(self.codeword_size//4, self.codeword_size//16,1)
        self.cort_conv4 = nn.Conv1d(self.codeword_size//16, self.codeword_size//32,1)
        self.cort_conv5 = nn.Conv1d(self.codeword_size//32, 1,1)

        # Trabecular
        self.trab_mlp1 = nn.Linear(self.feature_size + codeword_size, self.codeword_size,1)
        self.trab_conv2 = nn.Conv1d(self.codeword_size, self.codeword_size//4,1)
        self.trab_conv3 = nn.Conv1d(self.codeword_size//4, self.codeword_size//16,1)
        self.trab_conv4 = nn.Conv1d(self.codeword_size//16, self.codeword_size//32,1)
        self.trab_conv5 = nn.Conv1d(self.codeword_size//32, 1,1)

    def forward(self, elems, encoded_features):
        # elems shape: (B, E, 31) - Original features
        # encoded_features shape: (B, E, 16) - Encoded features
        B, E, _ = elems.size()

        # Combine elements with encoded features (B, E, 30 + codeword_size)
        x = torch.cat((elems[:, :, :-1], encoded_features), dim=2)

        # Extract cortical (xs == 1) and trabecular (xs == 0) features
        xs = elems[:, :, -1]  # cortical or trabecular indicator (last column in elems)
        
        cort_indices = (xs == 1)
        trab_indices = (xs == 0)
        x_combined = torch.zeros(B, E, 1, device=x.device)
        
        # Cortical
        x_cort = x[cort_indices]
        x_cort = self.act(self.cort_mlp1(x_cort))
        x_cort = x_cort.permute(1,0)
        x_cort = self.act(self.cort_conv2(x_cort))
        x_cort = self.act(self.cort_conv3(x_cort))
        x_cort = self.act(self.cort_conv4(x_cort))
        x_cort = self.act(self.cort_conv5(x_cort))
        x_cort = x_cort.permute(1,0)
        
        # Trabecular
        x_trab = x[trab_indices]
        x_trab = self.act(self.trab_mlp1(x_trab))
        x_trab = x_trab.permute(1,0)
        x_trab = self.act(self.trab_conv2(x_trab))
        x_trab = self.act(self.trab_conv3(x_trab))
        x_trab = self.act(self.trab_conv4(x_trab))
        x_trab = self.act(self.trab_conv5(x_trab))
        x_trab = x_trab.permute(1,0)
        
        
        # Concatenate cortical and trabecular back together
        x_combined[cort_indices] = x_cort
        x_combined[trab_indices] = x_trab

        return F.relu(x_combined)
