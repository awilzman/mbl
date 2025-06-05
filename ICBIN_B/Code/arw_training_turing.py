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
from geomloss import SamplesLoss

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
        distances_squared = torch.sum(diff ** 2, dim=-1)  # Shape: (B, N, M)
        
        min_distances1, _ = torch.min(distances_squared, dim=2)
        min_distances2, _ = torch.min(distances_squared, dim=1)
        mean1 = torch.mean(min_distances1)
        mean2 = torch.mean(min_distances2)

        r1 = cloud1.max(1).values - cloud1.min(1).values
        r2 = cloud2.max(1).values - cloud2.min(1).values
        
        # Return the max of the two
        return torch.max(mean1,mean2)


class GAN_Loss(nn.Module):
    def __init__(self):
        super(GAN_Loss, self).__init__()
        self.guess_loss = nn.BCELoss()
        
    def clip_gradients(self):
        # Iterate through all parameters and clip gradients
        for param in self.parameters():
            if param.grad is not None:
                nn.utils.clip_grad_norm_(param, self.clip_threshold)
    
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
    
class EMD_Loss(nn.Module):
    def __init__(self, p=2, blur=0.01, scaling=0.9):
        super().__init__()
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, scaling=scaling)

    def forward(self, p1, p2):
        # p1, p2: [B, N, 3]
        B, N, D = p1.shape
        loss = 0.0
        for i in range(B):
            loss += self.loss_fn(p1[i], p2[i])
        return loss / B
    
class Combined_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.chamfer = Chamfer_Loss()
        self.emd = EMD_Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, gt, pred):
        chamfer = self.chamfer(gt, pred)
        emd = self.emd(gt, pred)
        return self.alpha * chamfer + self.beta * emd
    
from scipy.optimize import linear_sum_assignment

class EMD_Loss_ND(nn.Module):
    def forward(self, p1, p2):
        # p1, p2: [B, N, 3]
        b, n, _ = p1.shape
        emd_total = 0.0

        for i in range(b):
            # Use torch to calculate distance on the same device (GPU or CPU)
            x = p1[i]
            y = p2[i]
            cost = torch.cdist(x, y, p=2)  # Euclidean distance (p=2)
            
            # Convert the cost matrix to NumPy for the linear sum assignment
            cost_np = cost.cpu().numpy()
            r, c = linear_sum_assignment(cost_np)
            
            # Accumulate the EMD for this batch
            emd_total += cost_np[r, c].mean()

        # Return the average EMD over the batch
        return torch.tensor(emd_total / b, dtype=torch.float32, device=p1.device)
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
                # R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                # R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                # R_xy = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                
                # rotate_x_bone = np.dot(sampled_bone, R_x)
                # rotate_y_bone = np.dot(sampled_bone, R_y)
                # rotate_xy_bone = np.dot(sampled_bone, R_xy)
                
                # self.data.append((rotate_x_bone, mt_no, side_int))
                # self.data.append((rotate_y_bone, mt_no, side_int))
                # self.data.append((rotate_xy_bone, mt_no, side_int))
    
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
            
            length, _ = batch_surf[:,:,0].max(dim=1)
            length = length.max()*2
            
            for _ in range(cycles):
                encoded_X, _ = network.encode(encoded_X)
                encoded_X = network.decode(encoded_X, batch_surf.shape[1])
            
            loss = loss_function(batch_surf, encoded_X)
            
            # import open3d as o3d
            # def set_point_cloud_color(point_cloud, color):
            #     point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(point_cloud.points), 1)))
            # color1 = np.array([0.229, 0.298, 0.512]) # blue = ground truth
            # color2 = np.array([0.728, 0.440, 0.145])
            
            # for i in range(batch_size):
            #     point_cloud1 = o3d.geometry.PointCloud()
            #     point_cloud1.points = o3d.utility.Vector3dVector(batch_surf[i].cpu().detach().numpy())
            #     set_point_cloud_color(point_cloud1, color=color1)
                
            #     point_cloud2 = o3d.geometry.PointCloud()
            #     point_cloud2.points = o3d.utility.Vector3dVector(encoded_X[i].cpu().detach().numpy())
            #     set_point_cloud_color(point_cloud2, color=color2)
                
            #     o3d.visualization.draw_geometries([point_cloud1,point_cloud2])
            
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

def train_vae(train_inputs, network, epochs, lr, wtdecay,
              batch_size, loss_fn, print_interval, device, num_points=1024):
    
    dataset = MTDataset(train_inputs, num_points)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    opt = optim.Adam(network.parameters(), lr=lr, weight_decay=wtdecay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
    
    losses = []
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        batch_losses = []
        
        for batch in loader:
            surf = batch['surf'].to(device)
            
            mu, logvar = network.encode(surf)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            recon = network.decode(z, surf.shape[1])
            
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            recon_loss = loss_fn(surf, recon)
            
            kl_wt = min(1.0, epoch / 100)  # KL annealing over first 100 epochs
            loss = recon_loss + kl_wt * 100 * kl
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            batch_losses.append(loss.item())
        
        avg_loss = torch.sqrt(torch.tensor(batch_losses).mean())
        losses.append(avg_loss.item())
        sched.step(avg_loss)
		
        if epoch % print_interval == 0:
            kl_ratio = 100 * kl / recon_loss
            t = time.time() - start
            print(f'Epoch {epoch:4d} | Loss: {avg_loss:.3e} | KL/recon %: {kl_ratio:.1f} | Time: {t:.1f}s')
			
        if time.time() - start > end_time:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {avg_loss:10.3e}, Time: {elapsed_time:7.1f}s')
            break
		
    return network, losses

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
            
            encoded_X, logvar = network.encode(encoded_X)
            kl = -0.5 * torch.sum(1 + logvar - encoded_X.pow(2) - logvar.exp(), dim=-1).mean()
            
            noise = torch.randn_like(encoded_X).to(device)
            noisy = sig_t * encoded_X + lamb_t * noise
            
            decoded_X = network.decode(noisy, batch_surf.shape[1])
            recon_loss = loss_function(batch_surf, decoded_X)
            
            kl_wt = min(1.0, epoch / 100)  # KL annealing over first 100 epochs
            loss = recon_loss + kl_wt * 100 * kl
            
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
        decoder_params = (list(Gnet.fold1.parameters()) + 
                          list(Gnet.trs_decoder.parameters()) + 
                          list(Gnet.fold2.parameters()))
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
            length, _ = data[:,:,0].max(dim=1)
            length = length.max()*2
            
            encoded, logvar = Gnet.encode(data)
            real_decoded = Gnet.decode(encoded,num_points)
            
            b_size = data.size(0)
            
            # Train Discriminator with real data
            label_real = torch.ones(b_size, device=device)
            output_real = Dnet(data).view(-1)
            lossD_real = loss_function(output_real, label_real)
            
            # Train Discriminator with fake data
            noise = torch.randn(b_size, 2, dec_hid, device=device)
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
    emd_losses = []
    
    with torch.no_grad():
        for X in loader:
            b = X['surf'].to(device)
            encoded_tensor, _ = model.encode(b)
            rec = model.decode(encoded_tensor, num_nodes)
            
            chamfer_loss = chamfer_loss_fn(b, rec)
            chamfer_losses.append(chamfer_loss.item())
            
            emd_loser = EMD_Loss_ND() #non differentiable, saves GPU memory
            emd_losses.append(emd_loser(b, rec))
        
        avg_chamfer_loss = torch.sqrt(torch.tensor(chamfer_losses).mean())
        avg_emd = sum(emd_losses) / len(chamfer_losses)
        
    return avg_chamfer_loss.item(), avg_emd
