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
        diff = cloud1.unsqueeze(2) - cloud2.unsqueeze(1) 
        distances_squared = torch.sum(diff ** 2, dim=-1)  
        
        # Find the mean minimum distance along the second and third dimensions
        min_distances1, _ = torch.min(distances_squared, dim=2)
        min_distances2, _ = torch.min(distances_squared, dim=1)
        
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
        
        # If predicted point cloud and target point cloud are provided, compute chamfer loss
        if pred_cloud is not None and tgt_cloud is not None:
            diff = pred_cloud.unsqueeze(2) - tgt_cloud.unsqueeze(1)  
            distances_squared = torch.sum(diff ** 2, dim=-1) 
            
            # Find the mean minimum distance along the second and third dimensions
            min_distances1, _ = torch.min(distances_squared, dim=2)
            min_distances2, _ = torch.min(distances_squared, dim=1)
            mean1 = torch.mean(min_distances1)
            mean2 = torch.mean(min_distances2)
            loss_cloud = max(mean1,mean2)
            loss += loss_cloud
        
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
    def __init__(self, file_paths, num_points=1024):
        self.file_paths = file_paths
        self.num_points = num_points
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
                
                self.data.append((sampled_bone, mt_no, side_int))
                
                # Generate rotated versions
                R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                R_xy = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                
                rotate_x_bone = np.dot(sampled_bone, R_x)
                rotate_y_bone = np.dot(sampled_bone, R_y)
                rotate_xy_bone = np.dot(sampled_bone, R_xy)
                
                self.data.append((rotate_x_bone, mt_no, side_int))
                self.data.append((rotate_y_bone, mt_no, side_int))
                self.data.append((rotate_xy_bone, mt_no, side_int))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    batch_x = [torch.FloatTensor(item[0]) for item in batch]
    batch_mt_no = torch.tensor([item[1] for item in batch], dtype=torch.int8)
    batch_side = torch.tensor([item[2] for item in batch], dtype=torch.int8)
    
    return {
        'surf': torch.stack(batch_x),
        'mt_no': batch_mt_no,
        'side': batch_side
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
            encoded_X = batch_surf.clone().to(device)
            
            for _ in range(cycles):
                encoded_X = network.encode(encoded_X)
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
            encoded_X = batch_surf.clone().to(device)
            
            # Encoding and decoding
            for _ in range(cycles):
                encoded_X = network.encode(encoded_X)
                latent = encoded_X.clone()
                encoded_X = network.decode(encoded_X, batch_surf.shape[1])
            
            # KL divergence loss
            mean, std = latent.mean(dim=1, keepdim=True), latent.std(dim=1, keepdim=True) + 1e-6
            kl_div = torch.distributions.kl.kl_divergence(
                torch.distributions.Normal(mean, std),
                torch.distributions.Normal(0, 1)
            ).mean()
            
            # Reconstruction loss
            recon_loss = loss_function(batch_surf, encoded_X)
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
            encoded_X = batch_surf.clone().to(device)
            
            lamb_t = max(min(noise_level / 10, 1), 0.05)
            sig_t = 1 - lamb_t
            
            encoded_X = network.encode(encoded_X)
            
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
    
    # Only include decoder parameters of Gnet
    if 'trs' in Gnet.__class__.__name__.lower():
        decoder_params = (list(Gnet.d_layers1.parameters()) + 
                          list(Gnet.trs_decoder.parameters()) + 
                          list(Gnet.d_layers2.parameters()))
    elif 'fold' in Gnet.__class__.__name__.lower():
        decoder_params = (list(Gnet.fold1.parameters()) + 
                          list(Gnet.fold2.parameters()))
    elif 'mlp' in Gnet.__class__.__name__.lower():
        decoder_params = (list(Gnet.d_layers1.parameters()) + 
                          list(Gnet.d_layers2.parameters()))
    else:
        decoder_params = Gnet.parameters()
        
                     
    optimizerG = optim.Adam(decoder_params, lr=learning_rate, weight_decay=decay)
    optimizerD = optim.Adam(Dnet.parameters(), lr=learning_rate, weight_decay=decay)
    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG,'min')
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD,'min')
    
    Gnet, Dnet = Gnet.to(device), Dnet.to(device)
    G_losses, D_losses = [], []
    
    start_time = time.time()
    max_time = abs(epochs) if epochs < 0 else None
    epochs = int(1e9) if epochs < 0 else epochs
    
    for epoch in range(1, epochs + 1):
        epoch_G_loss, epoch_D_loss = [], []
        
        for batch in train_loader:
            data = batch['surf'].to(device)
            encoded = Gnet.encode(data)
            real_decoded = Gnet.decode(encoded,num_points)
            
            b_size = data.size(0)
            
            # Train Discriminator with real data
            label_real = torch.ones(b_size, device=device)
            output_real = Dnet(data).view(-1)
            lossD_real = loss_function(output_real, label_real)
            
            # Train Discriminator with fake data
            noise = torch.randn(b_size, dec_hid, device=device)
            fake = Gnet.decode(noise, num_points)
            label_fake = torch.zeros(b_size, device=device)
            output_fake = Dnet(fake.detach()).view(-1)
            lossD_fake = loss_function(output_fake, label_fake)
            
            # Train Generator to fool Discriminator
            output_fake = Dnet(fake).view(-1)
            lossG = loss_function(output_fake, label_real, real_decoded, data)
            
            # Update Generator
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            # Update Discriminator
            optimizerD.zero_grad()
            (lossD_real + lossD_fake).backward()
            optimizerD.step()
            
            # Track losses
            epoch_D_loss.append((lossD_real + lossD_fake).item())
            epoch_G_loss.append(lossG.item())
        
        # Record mean losses for each epoch
        G_losses.append(sum(epoch_G_loss) / len(epoch_G_loss))
        D_losses.append(sum(epoch_D_loss) / len(epoch_D_loss))
        schedulerG.step(epoch_G_loss[-1])
        schedulerD.step(epoch_D_loss[-1])
        # Periodic logging
        if epoch % print_interval == 0 or (max_time and time.time() - start_time > max_time):
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch:4d}, Loss_D: {D_losses[-1]:.4f}, Loss_G: {G_losses[-1]:.4f}, Time: {elapsed:7.1f}s')
        
        if max_time and time.time() - start_time > max_time:
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
            encoded_tensor = model.encode(b)
            
            rec = model.decode(encoded_tensor, num_nodes)
            
            chamfer_loss = chamfer_loss_fn(b, rec)
            chamfer_losses.append(chamfer_loss.item())
            
            jsd_loser = JSD_Loss()
            jensen_shannon_divergences.append(jsd_loser(b, rec))
        
        avg_chamfer_loss = torch.sqrt(torch.tensor(chamfer_losses).mean())
        avg_jsd = sum(jensen_shannon_divergences) / len(chamfer_losses)
        
    return avg_chamfer_loss.item(), avg_jsd