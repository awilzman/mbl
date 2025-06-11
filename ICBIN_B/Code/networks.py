# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:03:37 2024

@author: Andrew R Wilzman
Structure https://github.com/qinglew/FoldingNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class jarvis(nn.Module):  # Discriminator network
    def __init__(self, insize):
        super(jarvis, self).__init__()
        self.k = 16
        
        # Convolutional layers with BatchNorm
        h3 = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, h3, 1),
            nn.BatchNorm1d(h3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(h3, h3, 1),
            nn.BatchNorm1d(h3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(h3, insize, 1),
            nn.BatchNorm1d(insize),
            nn.ReLU()
        )
        
        # Fully connected layers with BatchNorm
        self.fc_encoder2 = nn.Sequential(
            nn.Linear(insize, insize // 2),
            nn.BatchNorm1d(insize // 2),
            nn.ReLU(),
            nn.Linear(insize // 2, insize // 4),
            nn.BatchNorm1d(insize // 4),
            nn.ReLU(),
            nn.Linear(insize // 4, insize // 8),
            nn.BatchNorm1d(insize // 8),
            nn.ReLU(),
            nn.Linear(insize // 8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        b, n, c = data.size()

        # Compute k-NN and covariance
        knn_idx = knn(data, k=self.k)  # Custom k-NN function
        knn_x = index_points(data, knn_idx)  # (B, N, k, C)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        cov = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1)  # Covariance matrix flattened

        # Concatenate original data and covariance
        x = torch.cat([data, cov], dim=2).permute(0, 2, 1)  # Shape (B, C, N)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv4(x)

        # Global max pooling
        x = torch.max(x, dim=-1)[0]  # Shape (B, insize)

        # Fully connected layers
        x = self.fc_encoder2(x)

        return x

    
def knn(x, k):
    """
    Compute k-nearest neighbors for each point in a point cloud.
    https://github.com/qinglew/FoldingNet/blob/master/utils.py
    Parameters
    ----------
        x: Tensor of shape (B, C, N), input points data.
        k: int, number of nearest neighbors.

    Returns
    -------
        Tensor of shape (B, N, k), indices of k nearest neighbors.
    """
    sq_sum = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
    pairwise_distances = sq_sum - 2 * torch.matmul(x, x.transpose(2, 1)) + sq_sum.transpose(2, 1)  # (B, N, N)

    # Extract indices of the k nearest neighbors
    idx = pairwise_distances.topk(k=k, dim=-1, largest=False)[1]  # (B, N, k)
    return idx

def index_points(point_clouds, idx):
    """
    Index sub-tensors from a batch of tensors.

    Parameters
    ----------
        point_clouds: Tensor of shape [B, N, C], input points data.
        idx: Tensor of shape [B, N, k], sample index data.

    Returns
    -------
        Tensor of indexed points data with shape [B, N, k, C].
    """
    batch_size = point_clouds.shape[0]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=point_clouds.device).view(-1, 1, 1)
    new_points = point_clouds[batch_indices, idx, :]
    return new_points

class GraphLayer(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, N, C)
        """
        # KNN
        knn_idx = knn(x, k=self.k)
        knn_x = index_points(x, knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0]  # (B, N, C)
        
        # Feature Map
        x=x.permute(0,2,1)
        x = F.relu(self.bn(self.conv(x)))
        x=x.permute(0,2,1)
        return x

class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x
    
class arw_FoldingNet(nn.Module):
    def __init__(self, h1, h3):
        super(arw_FoldingNet, self).__init__()
        
        self.activate = nn.LeakyReLU()
        self.h1 = h1
        self.h3 = h3
        self.k = 16
        
        self.conv1 = nn.Conv1d(12, h3, 1)
        self.conv2 = nn.Conv1d(h3, h3, 1)
        self.conv3 = nn.Conv1d(h3, h3, 1)

        self.bn1 = nn.BatchNorm1d(h3)
        self.bn2 = nn.BatchNorm1d(h3)
        self.bn3 = nn.BatchNorm1d(h3)
        
        self.graph_encoder1 = GraphLayer(h3, h3 * 2)
        self.graph_encoder2 = GraphLayer(h3 * 2, h1)
        
        self.graph_encoder3 = GraphLayer(h3, h3 * 2)
        self.graph_encoder4 = GraphLayer(h3 * 2, h1)
        
        self.conv4 = nn.Conv1d(h1, h1, 1)
        self.bn4 = nn.BatchNorm1d(h1)
        
        self.conv5 = nn.Conv1d(h1, h1, 1)
        self.bn5 = nn.BatchNorm1d(h1)
        
        self.scale_mlp = nn.Sequential(
            nn.Linear(h1, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # scale for x, y, z
            nn.Sigmoid()
        )

        self.fold1 = FoldingLayer(h1 + 2, [h1, h1, 3])
        self.fold2 = FoldingLayer(h1 + 3, [h1, h1, 3])
        
        self.to_mu = nn.Linear(h1, h1)
        self.to_logvar = nn.Linear(h1, h1)
    
    def encode(self, data):
        b,n,c=data.size()
        
        knn_idx = knn(data, k=self.k)
        knn_x = index_points(data, knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        batch_size, num_nodes, num_features = data.size()
        
        cov = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1)
        x = torch.cat([data, cov], dim=2)
        x=x.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=x.permute(0,2,1)

        # two consecutive graph layers
        x1 = self.graph_encoder1(x)
        x1 = self.graph_encoder2(x1)
        x1=x1.permute(0,2,1)
        x1 = self.bn4(self.conv4(x1))
        
        x1 = torch.max(x1, dim=-1)[0]
        
        x2 = self.graph_encoder3(x)
        x2 = self.graph_encoder4(x2)
        x2=x2.permute(0,2,1)
        x2 = self.bn5(self.conv5(x2))
        
        x2 = torch.max(x2, dim=-1)[0]
        
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        
        h = torch.cat((x1,x2),dim=1)
        
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        
        return mu, logvar

    def decode(self, x, num_nodes):
        b, n, c = x.shape  # (B, 2, D
        
        # Generate a common grid
        base_x = torch.linspace(-40, 40, 50, device=x.device)  # normalized range
        base_y = torch.linspace(-60, 60, 50, device=x.device)
        X, Y = torch.meshgrid(base_x, base_y, indexing='ij')
        grid = torch.stack((X, Y), dim=0).view(2, -1)  # shape: (2, N)
        
        # Scale grid per batch
        grid = grid.unsqueeze(0).repeat(b, 1, 1)  # (B, 2, N)
        
        # Expand latent codes
        g_n = grid.shape[-1]
        x1 = x[:, 0, :].unsqueeze(2).repeat(1, 1, g_n)
        x2 = x[:, 1, :].unsqueeze(2).repeat(1, 1, g_n)
        
        # Folding
        r1 = self.fold1(grid, x1)
        r2 = self.fold2(r1, x2)
        
        # Random sampling
        idx = torch.randperm(g_n)[:num_nodes]
        r2 = r2.permute(0, 2, 1)[:, idx, :]  # (B, N, D)

        s = self.scale_mlp(x[:, 1, :]).unsqueeze(1)
        
        return r2*s
        
    def forward(self,x):
        b,n,c=x.size()
        y = self.encode(x)
        return self.decode(y,n)
        

class arw_TRSNet(nn.Module):
    def __init__(self, h1,h3):
        super(arw_TRSNet, self).__init__()
        self.activate = nn.SELU()
        self.h1 = h1
        while self.h1%16 != 0:
            self.h1 += 1
            print(f'changing h1 to {self.h1}')
        self.h4 = max(1,self.h1//16)
        
        self.k = 16
        
        self.conv1 = nn.Conv1d(12, h3, 1)
        self.conv2 = nn.Conv1d(h3, h3, 1)
        self.conv3 = nn.Conv1d(h3, h1, 1)
        self.conv4 = nn.Conv1d(h1, h1, 1)

        self.bn1 = nn.BatchNorm1d(h3)
        self.bn2 = nn.BatchNorm1d(h3)
        self.bn3 = nn.BatchNorm1d(h1)
        self.bn4 = nn.BatchNorm1d(h1)
        
        self.trs_encoder = nn.TransformerEncoderLayer(self.h1,self.h4,batch_first=True)
        
        self.trs_decoder = nn.TransformerDecoderLayer(self.h1, self.h4, batch_first=True)
        
        xx = np.linspace(-40, 40, 50, dtype=np.float32)
        yy = np.linspace(-60, 60, 50, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(h1 + 2, [h1, h1, 3])
        self.fold2 = FoldingLayer(h1 + 3, [h1, h1, 3])
        
        
    def encode(self, data):
        b,n,c=data.size()
        knn_idx = knn(data, k=self.k)
        knn_x = index_points(data, knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        batch_size, num_nodes, num_features = data.size()
        cov = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1)
        x = torch.cat([data, cov], dim=2)
        
        x=x.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=x.permute(0,2,1)

        # two consecutive graph layers
        x = self.trs_encoder(x)
        x=x.permute(0,2,1)
        x = self.bn4(self.conv4(x))
        
        x = torch.max(x, dim=-1)[0]
        return x

    def decode(self, x, num_nodes):
        b,c = x.shape
        
        x = self.trs_decoder(x,x)
        
        grid = self.grid.to(x.device)
        grid = grid.unsqueeze(0).repeat(b, 1, 1)
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        rand_ind = torch.randperm(grid.shape[-1])[:num_nodes]
        recon2 = recon2.permute(0,2,1)[:,rand_ind,:]
        
        return recon2
