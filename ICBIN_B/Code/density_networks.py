# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:45:17 2024
Updated 04/13/2025
@author: Andrew
v6.0

Autoencoder for 2D sequence data with positional encoding and FiLM conditioning.

This model processes temporal or sequential input features (tetrahedral element data) 
using a convolutional encoder-decoder architecture. It incorporates:

- FiLM (Feature-wise Linear Modulation) layers to condition on a header vector `h`
- Learnable attention pooling with a global query vector
- Sinusoidal positional embeddings in both encoder and decoder

Main Components:
.encode(x, h):
    - Sorts input sequence spatially
    - Applies convolutions and FiLM modulation
    - Performs attention pooling to produce latent representation z

.decode(z, h, T):
    - Expands z into a temporal sequence using positional encoding
    - Applies FiLM modulation and convolutional decoding
    - Outputs reconstructed sequence of original length T

.forward(x, h):
    - Applies encode followed by decode
    - Used for end-to-end forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class tet10_autoencoder(nn.Module):
    def __init__(self, h=128, out_c=1):
        super().__init__()
        # Parameter setup
        self.in_c  = 31 + out_c
        self.h_c   = h * 2
        self.l_c   = h
        self.out_c = out_c
        self.act   = nn.LeakyReLU()
        
        # encoder
        self.e1 = nn.Conv1d(self.in_c, self.h_c, 8, padding=5)
        self.e2 = nn.Conv1d(self.h_c, self.l_c, 3, padding=2)
        self.film = nn.Linear(7, self.l_c * 2)
        
        # attention pooling
        self.q = nn.Parameter(torch.randn(1, self.l_c))  # global query

        # decoder
        self.film_dec = nn.Linear(7, self.l_c * 2)
        self.d1 = nn.Conv1d(self.l_c, self.h_c, 8, padding=5)
        self.d2 = nn.Conv1d(self.h_c, self.out_c, 4, padding=2)
        
        self.out_mlps = nn.ModuleList([
        nn.Sequential(
            nn.Conv1d(1, h//4, 1),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv1d(h//4, h//2, 1),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv1d(h//2, h, 1),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv1d(h, 4, 1),
            nn.LeakyReLU(),
            nn.Conv1d(4, 1, 1)
        ) for _ in range(self.out_c)
    ])

    def encode(self, x, h):  # x: (B,T,C), h: (B,7)
        # Sort elements
        key = x[:,:, 0]*1e6 + x[:,:, 1]*1e3 + x[:,:, 2]
        idxs = torch.argsort(key, dim=1)
        x = torch.gather(x, 1, idxs.unsqueeze(-1).expand(-1, -1, x.size(2)))

        # Position embedding 
        y = x.permute(0,2,1)  # (B,C,T)
        pe = self.get_pos_emb(y.size(2), y.size(1), y.device)
        y = y + pe

        # Convolutions
        y = self.act(self.e1(y))
        y = self.act(self.e2(y))  # (B,D,T)
        y = y.permute(0,2,1)      # (B,T,D)

        # Inject header embedding with FiLM
        film_params = self.film(h)  
        g, b = film_params.chunk(2, dim=-1)
        y=g*y+b

        # Attention pooling
        q = self.q.expand(x.size(0), 1, -1)  # (B,1,D)
        attn = (q @ y.transpose(1,2)) / (self.l_c ** 0.5)
        attn = attn.softmax(dim=-1)
        z = attn @ y  # (B,1,D)
        
        return z.squeeze(1)                                # (B,D)

    def get_pos_emb(self, T, D, dev):
        #Position embedding function
        pos = torch.arange(T, device=dev).unsqueeze(1)
        i   = torch.arange(D, device=dev).unsqueeze(0)
        rates = 1 / (10000 ** ((i//2)*2/D))
        ang = pos * rates
        pe = torch.zeros_like(ang)
        pe[:,0::2] = torch.sin(ang[:,0::2])
        pe[:,1::2] = torch.cos(ang[:,1::2])
        return pe.T.unsqueeze(0)  # (1,D,T)

    def unfold(self, z, T):
        # Unfold codeword to sequence using position embedding
        B,D = z.shape
        z2 = z.unsqueeze(-1).expand(-1,-1,T)       # (B,D,T)
        pe = self.get_pos_emb(T, D, z.device).expand(B,-1,-1)
        return z2 + pe

    def decode(self, z, h, T):
        seq = self.unfold(z, T)  # (B,D,T)

        # Inject header embedding with FiLM
        film_params = self.film_dec(h)
        g, b = film_params.chunk(2, dim=-1)
        seq=g*seq.transpose(1,2)+b

        #Convolutions 
        y = self.act(self.d1(seq.transpose(1,2)))
        y = self.act(self.d2(y))
        y = y.permute(0,2,1)[:,:T,:] #Restrict output
        
        out = []
        for i in range(self.out_c):
            ch = y[:, :, i].unsqueeze(1)  # (B,1,T)
            ch = self.out_mlps[i](ch)     # (B,1,T)
            out.append(ch)
            
        return torch.cat(out,dim=1).permute(0,2,1)

    def forward(self, x, h):
        B,T,_ = x.shape
        z = self.encode(x, h)
        return self.decode(z, h, T)

