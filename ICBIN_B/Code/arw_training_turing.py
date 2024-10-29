# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# functions:
#    MTDataset(Dataset): Data loader 
#       this is used to fish for data as needed instead of loading in 
#       all of the pointclouds at once.
#       only the filepath matrix is needed to tag data for batching
# =============================================================================
import torch
from torch import nn
import numpy as np
import time
import h5py
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
#%% Loss Functions
class Chamfer_Loss(nn.Module):
    def __init__(self, clip_threshold=1.0):
        super(Chamfer_Loss, self).__init__()
        self.clip_threshold = clip_threshold

    def forward(self, inputs, preds):
        return self.compute_chamfer_loss(inputs, preds)
                
    def compute_chamfer_loss(self, inputs, preds):
        return self.compute_loss(inputs, preds)
    
    def clip_gradients(self):
        # Iterate through all parameters and clip gradients
        for param in self.parameters():
            if param.grad is not None:
                nn.utils.clip_grad_norm_(param, self.clip_threshold)

    def compute_loss(self, cloud1, cloud2):
        # Compute squared Euclidean distances using broadcasting
        diff = cloud1.unsqueeze(2) - cloud2.unsqueeze(1)  # Shape: (batch_size, num_points_cloud1, num_points_cloud2, 3)
        distances_squared = torch.sum(diff ** 2, dim=-1)  # Shape: (batch_size, num_points_cloud1, num_points_cloud2)
        
        # Find the mean minimum distance along the second and third dimensions
        min_distances1, _ = torch.min(distances_squared, dim=2)  # Shape: (batch_size, num_points_cloud1)
        min_distances2, _ = torch.min(distances_squared, dim=1)  # Shape: (batch_size, num_points_cloud2)
        
        # Compute the Chamfer loss as the sum of mean minimum distances
        mean1 = torch.mean(min_distances1)
        mean2 = torch.mean(min_distances2)
        
        return max(mean1,mean2)

class GAN_Loss(nn.Module):
    def __init__(self):
        super(GAN_Loss, self).__init__()
        self.guess_loss = nn.BCELoss()
    
    def forward(self, predicted, actual, pred_cloud=None, tgt_cloud=None):
        loss = self.guess_loss(predicted, actual)
        
        # If predicted point cloud and target point cloud are provided, compute KL divergence loss
        if pred_cloud is not None and tgt_cloud is not None:
            loss_cloud = Chamfer_Loss.compute_loss(self,pred_cloud, tgt_cloud)
            loss += loss_cloud/100
        
        return loss
    
class JSD_Loss(nn.Module):
    def __init__(self):
        super(JSD_Loss,self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        # Compute probabilities from logits using softmax
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        # Calculate the average probability distribution
        m = 0.5 * (net_1_probs + net_2_probs)

        # Initialize loss
        loss = 0.0

        # Calculate Kullback-Leibler (KL) divergence for both networks
        # and accumulate the loss
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean")

        return 0.5 * loss

#%% Datasets
class MTDataset(Dataset):
    def __init__(self, file_paths, num_points=1024, k=16):
        self.file_paths = file_paths
        self.num_points = num_points
        self.k = k
        self.data = []
        
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as hf:
                bone = hf['Surface'][:]
                mt_no = int(np.array(hf['MTno']))
                side = np.array(hf['Side'])
                side_int = 0 if 'L' in str(side) else 1
                
                num_samples = min(self.num_points, bone.shape[0])
                sampled_indices = torch.randperm(bone.shape[0])[:num_samples]
                sampled_bone = bone[sampled_indices]
                
                edge_index = self.compute_knn_graph(sampled_bone, k=self.k)
                
                self.data.append((sampled_bone, mt_no, side_int, edge_index))
                
                # Generate extra samples by rotating 180 degrees on x and y axes
                R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                R_xy = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                
                rotate_x_bone = np.dot(sampled_bone, R_x)
                rotate_y_bone = np.dot(sampled_bone, R_y)
                rotate_xy_bone = np.dot(sampled_bone, R_xy)
                
                edge_index_rotate_x = self.compute_knn_graph(rotate_x_bone, k=self.k)
                edge_index_rotate_y = self.compute_knn_graph(rotate_y_bone, k=self.k)
                edge_index_rotate_xy = self.compute_knn_graph(rotate_xy_bone, k=self.k)
                
                self.data.append((rotate_x_bone, mt_no, side_int, edge_index_rotate_x))
                self.data.append((rotate_y_bone, mt_no, side_int, edge_index_rotate_y))
                self.data.append((rotate_xy_bone, mt_no, side_int, edge_index_rotate_xy))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def compute_knn_graph(self, positions, k):
        neighbors = NearestNeighbors(n_neighbors=k+1).fit(positions)
        distances, indices = neighbors.kneighbors(positions)
        edge_index = []
        
        for i in range(positions.shape[0]):
            for j in range(1, k+1):  # Start from 1 to skip the point itself
                edge_index.append([i, indices[i, j]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

def custom_collate(batch):
    batch_x = [torch.FloatTensor(item[0]) for item in batch]
    batch_mt_no = torch.tensor([item[1] for item in batch], dtype=torch.int8)
    batch_side = torch.tensor([item[2] for item in batch], dtype=torch.int8)
    batch_edge_index = []

    num_nodes = [item[0].shape[0] for item in batch]
    cumsum_nodes = np.cumsum([0] + num_nodes)
    
    for i, item in enumerate(batch):
        edge_index = item[3] + cumsum_nodes[i]
        batch_edge_index.append(edge_index)
    
    batch_edge_index = torch.cat(batch_edge_index, dim=1)
    
    return {
        'surf': torch.stack(batch_x),
        'mt_no': batch_mt_no,
        'side': batch_side,
        'knn': batch_edge_index
    }
#%% Training functions
# Please add saving function to report model parameters during training
def train_autoencoder(training_inputs, network, epochs, learning_rate, wtdecay, 
                      batch_size, loss_function, print_interval, device,  
                      num_points=1024,cycles=1):
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
        
    track_losses = []
    start = time.time()
    print('Timer started')
    
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_surf = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            encoded_X = batch_surf.clone().to(device)
            
            for _ in range(cycles):
                encoded_X = network.encode(encoded_X, edge_index)
                encoded_X = network.decode(encoded_X, batch_surf.shape[1])
            
            loss = loss_function(batch_surf, encoded_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store batch loss for epoch aggregation
            epoch_losses.append(loss.item())
        
        # Compute average epoch loss
        training_loss = torch.mean(torch.tensor(epoch_losses))
        track_losses.append(training_loss.item())
        scheduler.step(training_loss)
        
        if epoch % print_interval == 0:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
        
        if time.time() - start > end_time:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
            break
        
    return network, epoch_losses

def train_vae(training_inputs, network, epochs, learning_rate, wtdecay,
              batch_size, loss_function, print_interval, device, num_points=1024, cycles=1):
    
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    max_duration = abs(epochs) if epochs < 0 else None
    epochs = int(1e9) if epochs < 0 else epochs

    track_losses = []
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        
        for batch in train_loader:
            batch_surf = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            encoded_X = batch_surf
            
            # Encoding and decoding
            for _ in range(cycles):
                encoded_X = network.encode(encoded_X, edge_index)
            latent = encoded_X.clone()
            decoded_X = network.decode(latent, batch_surf.shape[1])
            
            # KL divergence loss
            mean, std = latent.mean(dim=2, keepdim=True), latent.std(dim=2, keepdim=True) + 1e-6
            kl_div = torch.distributions.kl.kl_divergence(
                torch.distributions.Normal(mean, std),
                torch.distributions.Normal(0, 1)
            ).mean()
            
            # Reconstruction loss
            recon_loss = loss_function(batch_surf, decoded_X)
            loss = recon_loss + kl_div * 100
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        training_loss = torch.sqrt(torch.tensor(epoch_losses).mean())
        track_losses.append(training_loss.item())
        scheduler.step(training_loss)
        
        # Printing and time-based stopping
        if epoch % print_interval == 0 or (max_duration and time.time() - start > max_duration):
            klrecon = 100*kl_div/recon_loss
            print(f'Epoch: {epoch:4d}, Loss: {training_loss:.3e}, KL/recon %: {klrecon:.1f}, Time: {time.time() - start:.1f}s')
        
        if max_duration and time.time() - start > max_duration:
            break
    
    return network, track_losses

def train_diffusion(training_inputs, network, epochs, learning_rate, wtdecay,
                    batch_size, loss_function, print_interval, device, noise_level=1, num_points=1024, cycles=1):
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
        
    track_losses = []
    start = time.time()
    print('Timer started')
    
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch_surf = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            encoded_X = batch_surf.clone().to(device)
            
            lamb_t = max(min(noise_level / 10, 1), 0.05)
            sig_t = 1 - lamb_t
            
            encoded_X = network.encode(encoded_X, edge_index)
            
            noise = torch.randn_like(encoded_X).to(device)
            noisy = sig_t * encoded_X + lamb_t * noise
            
            encoded_X = network.decode(noisy, batch_surf.shape[1])
            loss = loss_function(batch_surf, encoded_X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        training_loss = torch.sqrt(torch.tensor(epoch_losses).mean())
        track_losses.append(training_loss.item())
        scheduler.step(training_loss)
        if epoch % print_interval == 0 or time.time() - start > end_time:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
        
        if time.time() - start > end_time:
            break
        
    return network, track_losses

# GAN training function
def train_GD(training_inputs, Gnet, Dnet, dec_hid, epochs, learning_rate, decay,
             batch_size, loss_function, print_interval, device, num_points=1024):
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=custom_collate)
    
    optimizerG = torch.optim.Adam(Gnet.parameters(), lr=learning_rate, weight_decay=decay)
    optimizerD = torch.optim.Adam(Dnet.parameters(), lr=learning_rate, weight_decay=decay)
    
    G_losses = []
    D_losses = []
    
    Gnet = Gnet.to(device)
    Dnet = Dnet.to(device)
    
    start = time.time()
    print('Timer started')
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
    
    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            data = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            
            b_size = len(data)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            
            # Train Discriminator with real data
            output = Dnet(data).view(-1)
            lossD_real = loss_function(output, label)
            optimizerD.zero_grad()
            lossD_real.backward()
            optimizerD.step()
            # Train Discriminator with fake data
            b = Gnet.encode(data, edge_index)
            noise = torch.randn(b_size, b.shape[1], b.shape[2], device=device)
            fake = Gnet.decode(noise, data.shape[1])
            label.fill_(0)
            output = Dnet(fake).view(-1)
            lossD_fake = loss_function(output, label)
            optimizerD.zero_grad()
            lossD_fake.backward()
            optimizerD.step()
            
            # Train Generator
            fake = Gnet.decode(noise, data.shape[1])
            label.fill_(1)
            output = Dnet(fake).view(-1)
            lossG = loss_function(output, label, fake, data)
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            # Track losses
            lossD = lossD_real + lossD_fake
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
        if epoch % print_interval == 0 or time.time() - start > end_time:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Loss_D: {lossD.item():.4f}, Loss_G: {lossG.item():.4f}, Time: {elapsed_time:7.1f}s')
        
        if time.time() - start > end_time:
            break
    
    return Gnet, G_losses, Dnet, D_losses

def model_eval_chamfer(x, model, num_nodes, device, batch_size=10):
    ds = MTDataset(x, num_nodes)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=custom_collate)
    
    chamfer_loss_fn = Chamfer_Loss()
    chamfer_losses = []
    jensen_shannon_divergences = []
    
    with torch.no_grad():
        for X in loader:
            b = X['surf'].to(device)
            k = X['knn'].to(device)
            encoded_tensor = model.encode(b, k)
            
            rec = model.decode(encoded_tensor, num_nodes)
            
            chamfer_loss = chamfer_loss_fn(b, rec)
            chamfer_losses.append(chamfer_loss.item())
            
            jsd_loser = JSD_Loss()
            jensen_shannon_divergences.append(jsd_loser(b, rec))
        
        avg_chamfer_loss = torch.sqrt(torch.tensor(chamfer_losses).mean())
        avg_jsd = sum(jensen_shannon_divergences) / len(chamfer_losses)
        
    return avg_chamfer_loss.item(), avg_jsd