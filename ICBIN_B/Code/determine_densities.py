# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:43:18 2024

@author: arwilzman
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import h5py
from torch.utils.data import Dataset, DataLoader

class MetatarsalDataset(Dataset):
    def __init__(self, h5_file):
        with h5py.File(h5_file, 'r') as file:
            self.data = {key: torch.tensor(file[key]) for key in file.keys()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]
        return self.data[key]

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
        # Output size depends on whether the GRU is bidirectional
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 30)

    def forward(self, x, lengths):
        # Pack padded sequences for variable-length inputs
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # Unpack the sequence

        last_hidden_states = output[torch.arange(output.size(0)), lengths - 1, :]
        encoded = self.fc(last_hidden_states)

        return encoded
    
class tet10_decoder(nn.Module):
    def __init__(self, num_ele, codeword_size=16):
        super(tet10_decoder, self).__init__()
        self.num_ele = num_ele
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        # Define the network
        self.fc1 = nn.Linear(self.codeword_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.num_ele * self.feature_size)
        
    def forward(self, x):
        # x shape: (B, 1, 16)
        x = x.squeeze(1)  # Remove the dimension with size 1 (B, 16)
        
        x = self.fc1(x)  # (B, 16) -> (B, 64)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)  # (B, 64) -> (B, 128)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)  # (B, 128) -> (B, E * 30)
        
        # Reshape to (B, E, 30)
        x = x.view(-1, self.num_ele, self.feature_size)
        
        return x
    
class tet10_densify(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_densify, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        # Define the network layers
        self.fc1 = nn.Linear(self.feature_size + self.codeword_size, 128)  # Combine input features and codeword
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)  # Output size is 1 (density)
        
    def forward(self, x, encoded_features):
        # x shape: (B, E, 30) - Original features
        # encoded_features shape: (B, 1, 16) - Encoded features
        
        # Repeat encoded_features for each element
        encoded_features_repeated = encoded_features.expand(-1, x.size(1), -1)  # (B, E, 16)
        
        # Concatenate the original features with the repeated encoded features
        combined = torch.cat((x, encoded_features_repeated), dim=2)  # (B, E, 30 + 16)
        
        # Pass through fully connected layers
        x = self.fc1(combined)  # (B, E, 30 + 16) -> (B, E, 128)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)  # (B, E, 128) -> (B, E, 64)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)  # (B, E, 64) -> (B, E, 1)
        
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str,default='')
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('-e','--epochs', type=int,default=0)
    parser.add_argument('-t','--traintime', type=int,default=10)
    parser.add_argument('-lr', type=float,default=1e-3)
    parser.add_argument('--decay', type=float,default=1e-6)
    parser.add_argument('--chpt',type=int,default=0)
    parser.add_argument('--chkdecay',type=float,default=0.95)
    parser.add_argument('--batch', type=int,default=1)
    parser.add_argument('--pint', type=int,default=0)
    parser.add_argument('--noise', type=int,default=3)
    parser.add_argument('--name', type=str,default='')
    parser.add_argument('--load', type=str,default='')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('-a','--autoencode', action='store_true')
    parser.add_argument('-vae', action='store_true') #variational AE
    parser.add_argument('-v','--visual', action='store_true') #visualize
    
    args = parser.parse_args(['--direct','../','-n','fold',
                              '-a',
                              '-lr','1e-4','--decay','1e-5',
                              '-e','0',
                              '-t','600',
                              '--pint','1',
                              '--chpt','0',
                              '--cycles','1',
                              '--noise','5',
                              '--name','',
                              '--load',''])
                    
    #Initialize vars
    if args.load != '':
        if args.load[-4:] != '.pth': # must be .pth
            args.load += '.pth'
            
    