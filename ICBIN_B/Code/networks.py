# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:03:37 2024

@author: Andrew R Wilzman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class jarvis(nn.Module): # Discriminator network
    def __init__(self, insize):
        super(jarvis, self).__init__()
        self.activate = nn.SELU()
        self.h1 = 256
        self.h2 = 64
        self.h3 = 32
        self.h4 = 16
        # First encoding MLP
        self.fc_encoder1 = nn.Sequential(
            nn.Linear(3, self.h4),  
            self.activate,
            nn.Linear(self.h4, self.h3),
            self.activate,
            nn.Linear(self.h3, self.h2),
            nn.Dropout(0.2),
            nn.Linear(self.h2, self.h1),
            self.activate,
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.2),
            self.activate,
            nn.Linear(self.h2, self.h3),
            self.activate,
            nn.Linear(self.h3, self.h4),
            self.activate,
            nn.Linear(self.h4, 1),
            nn.Sigmoid())
        
        self.pooler = nn.AdaptiveAvgPool1d(self.h1)
        
        self.fc_encoder2 = nn.Sequential(
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.2),
            self.activate,
            nn.Linear(self.h2, self.h3),
            self.activate,
            nn.Linear(self.h3, self.h4),
            self.activate,
            nn.Linear(self.h4, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        y = self.fc_encoder1(x)
        y = self.pooler(y.permute(0,2,1))
        y = self.fc_encoder2(y)
        x = y.permute(0,2,1)
        return x

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
        self.activate = nn.ReLU()
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
        
        self.conv4 = nn.Conv1d(h1, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        
        xx = np.linspace(-40, 40, 45, dtype=np.float32)
        yy = np.linspace(-60, 60, 45, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(512 + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(512 + 3, [512, 512, 3])
    
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
        x = self.graph_encoder1(x)
        x = self.graph_encoder2(x)
        x=x.permute(0,2,1)
        x = self.bn4(self.conv4(x))
        
        x = torch.max(x, dim=-1)[0]
        return x

    def decode(self, x, num_nodes):
        b,c = x.shape
        
        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(b, 1, 1)  # (B, 2, 45 * 45)
        n = 45**2
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, 45 * 45)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        rand_ind = torch.randperm(n)[:num_nodes]
        recon2 = recon2.permute(0,2,1)[:,rand_ind,:]
        
        return recon2
    
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

class arw_TRSNet(nn.Module):
    def __init__(self, h1,h3,initial_state=None,max_depth=8):
        super(arw_TRSNet, self).__init__()
        self.activate = nn.SELU()
        self.h1 = h1
        while self.h1%16 != 0:
            self.h1 += 1
            print(f'changing h1 to {self.h1}')
        self.h4 = max(1,self.h1//16)
        
        self.input_dim = 3
        
        self.max_depth = max_depth
        self.max_width = self.h1
        
        self.e_layers1 = nn.ModuleList()
        self.e_layers2 = nn.ModuleList()
        self.d_layers1 = nn.ModuleList()
        self.d_layers2 = nn.ModuleList()
        
        self.initialize_state(initial_state)
        
        self.pooler1 = nn.AdaptiveMaxPool1d(self.h1)
        
        self.trs_encoder = nn.TransformerEncoderLayer(self.h1,self.h4,batch_first=True)        
        
        self.pooler2 = nn.AdaptiveMaxPool1d(1)
        # Result: 1 x h1 "codeword"
        
        self.trs_decoder = nn.TransformerDecoderLayer(self.h1, self.h4, batch_first=True)
        
    def initialize_state(self, initial_state):
        if initial_state is None:
            initial_state = [[(self.input_dim,self.h4),(self.h4,self.input_dim)],
                             [(self.input_dim*2,self.h1)],[(self.h1+2,3)],[(self.h1+3,3)]]
        for widths, layer_list in zip(initial_state, [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]):
            for insize, outsize in widths:
                layer_list.append(nn.Linear(insize, outsize))

    def add_layer(self, mlp_no, index):
        layer_list = self.get_layer_list(mlp_no)
        if len(layer_list) < self.max_depth:
            if index == 0:
                if mlp_no == 1:
                    prev_width = self.input_dim
                elif mlp_no == 2:
                    prev_width = self.input_dim*2
                elif mlp_no == 3:
                    prev_width = self.h1 + 2
                else:
                    prev_width = self.h1 + 3
            else:
                prev_width = layer_list[index - 1].out_features
            new_layer = nn.Linear(prev_width, prev_width)
            nn.init.xavier_uniform_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
            layer_list.insert(index, new_layer)
        else:
            # add width expansion
            print(f'maxed layer depth for MLP {mlp_no}')

    def change_width(self, mlp_no, index, new_width):
        layer_list = self.get_layer_list(mlp_no)
        new_width = min(new_width,self.max_width)
        if 0 <= index < len(layer_list)-1:
            prev_in = layer_list[index].in_features
            layer_list[index] = nn.Linear(prev_in, new_width)
    
            # Update subsequent layer
            if index < len(layer_list) - 1:
                next_width = layer_list[index + 1].out_features
                layer_list[index + 1] = nn.Linear(new_width, next_width)
                
    def get_layer_list(self, mlp_no):
        
        if mlp_no == 1:
            return self.e_layers1
        elif mlp_no == 2:
            return self.e_layers2
        elif mlp_no == 3:
            return self.d_layers1
        elif mlp_no == 4:
            return self.d_layers2
        else:
            return [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]
        
    def encode(self, x, knn=None):
        y = x.clone()
        
        for layer in self.e_layers1:
            y = self.activate(layer(y))
            
        y = torch.cat([x,y],dim=2)
        
        for layer in self.e_layers2:
            y = self.activate(layer(y))
        
        x = y.permute(0,2,1)
        x = self.pooler2(self.trs_encoder(self.pooler1(x)))
        x = x.permute(0,2,1)
        
        return x

    def decode(self, x, num_nodes):
        num_nodes_x = int(1+(num_nodes * 120 / 60) ** 0.5)
        num_nodes_y = int(1+(num_nodes * 60 / 120) ** 0.5)
        x_grid = torch.linspace(1, 120, num_nodes_x).to(x.device)
        y_grid = torch.linspace(1, 60, num_nodes_y).to(x.device)
        grid_points = torch.cartesian_prod(x_grid, y_grid)
        grid_points = grid_points[:num_nodes]
        y = grid_points.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = x.repeat(1,num_nodes,1)
        
        k = torch.cat([x,y], dim=2)
        
        for layer in self.d_layers1:
            k = self.activate(layer(k))
        
        x = self.trs_decoder(x, x)
        
        x = torch.cat([x,k],dim=2)
        
        for layer in self.d_layers2:
            x = self.activate(layer(x))
        
        return x
    
    def forward(self, input_data):
        
        x = self.encode(input_data)
        y = self.decode(x)

        return y

class arw_MLPNet(nn.Module):
    def __init__(self, h1,h3, initial_state=None,max_depth=8,max_width=1024):
        super(arw_MLPNet, self).__init__()
        self.activate = nn.SELU()
        
        self.h1 = h1
        self.h3 = h3
        
        self.input_dim = 3
        
        self.max_depth = max_depth
        self.max_width = max_width
        
        self.e_layers1 = nn.ModuleList()
        self.e_layers2 = nn.ModuleList()
        self.d_layers1 = nn.ModuleList()
        self.d_layers2 = nn.ModuleList()
        
        self.initialize_state(initial_state)
        
        self.pooler1 = nn.AdaptiveMaxPool1d(1)
        
        # Result: 1 x h1 "codeword"
        
    def initialize_state(self, initial_state):
        if initial_state is None:
            initial_state = [[(self.input_dim,self.h1)],[(self.h3,self.h1)],[(self.h1+2,3)],[(self.h1+3,3)]]
        for widths, layer_list in zip(initial_state, [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]):
            for insize, outsize in widths:
                layer_list.append(nn.Linear(insize, outsize))

    def add_layer(self, mlp_no, index):
        layer_list = self.get_layer_list(mlp_no)
        if len(layer_list) < self.max_depth:
            if index == 0:
                if mlp_no == 1:
                    prev_width = self.input_dim
                elif mlp_no == 2:
                    last = self.get_layer_list(1)
                    prev_width = last[-1].out_features
                elif mlp_no == 3:
                    prev_width = self.h1 + 2
                else:
                    prev_width = self.h1 + 3
            else:
                prev_width = layer_list[index - 1].out_features
            new_layer = nn.Linear(prev_width, prev_width)
            nn.init.xavier_uniform_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
            layer_list.insert(index, new_layer)
        else:
            # add width expansion
            print(f'maxed layer depth for MLP {mlp_no}')

    def change_width(self, mlp_no, index, new_width):
        layer_list = self.get_layer_list(mlp_no)
        new_width = min(new_width,self.max_width)
        if 0 <= index < len(layer_list)-1:
            prev_in = layer_list[index].in_features
            layer_list[index] = nn.Linear(prev_in, new_width)
    
            # Update subsequent layer
            if index < len(layer_list) - 1:
                next_width = layer_list[index + 1].out_features
                layer_list[index + 1] = nn.Linear(new_width, next_width)
                
    def get_layer_list(self, mlp_no):
        
        if mlp_no == 1:
            return self.e_layers1
        elif mlp_no == 2:
            return self.e_layers2
        elif mlp_no == 3:
            return self.d_layers1
        elif mlp_no == 4:
            return self.d_layers2
        else:
            return [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]
    
    def encode(self, x):
        y = x.clone()
        
        for layer in self.e_layers1:
            y = self.activate(layer(y))
            
        y = y.permute(0,2,1)
        y = self.pooler1(y)
        y = y.permute(0,2,1)
        
        for layer in self.e_layers2:
            y = self.activate(layer(y))
        
        return y

    def decode(self, x, num_nodes):
        
        num_nodes_x = int(1+(num_nodes * 120 / 60) ** 0.5)
        num_nodes_y = int(1+(num_nodes * 60 / 120) ** 0.5)
        x_grid = torch.linspace(1, 120, num_nodes_x).to(x.device)
        y_grid = torch.linspace(1, 60, num_nodes_y).to(x.device)
        grid_points = torch.cartesian_prod(x_grid, y_grid)
        grid_points = grid_points[:num_nodes]
        y = grid_points.unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        x = x.repeat(1,num_nodes,1)
        
        k = torch.cat([x,y], dim=2)
        
        for layer in self.d_layers1:
            k = self.activate(layer(k))
            
        x = torch.cat([x,k],dim=2)
        
        for layer in self.d_layers2:
            x = self.activate(layer(x))
        
        return x
    
    def forward(self, input_data, knn=None):
        
        x = self.encode(input_data)
        y = self.decode(x)

        return y