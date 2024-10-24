# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:46:23 2024

@author: Andrew
v3.0
"""
import torch
import torch.nn as nn
import h5py
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import argparse
import time
import pandas as pd
import density_networks as dnets

class MetatarsalDataset(Dataset):
    def __init__(self, h5_file):
        with h5py.File(h5_file, 'r') as file:
            self.data = []
            self.labels = []
            self.lengths = []
            for key in file.keys():
                full_data = torch.tensor(file[key][:], dtype=torch.float32)
                features = full_data[:, :-1]
                labels = full_data[:, -1]
                self.data.append(features)
                self.labels.append(labels)
                self.lengths.append(features.size(0))

        # Automatically calculate the scaling factor based on the maximum label value
        all_labels = torch.cat(self.labels)
        self.label_scaling_factor = all_labels.max().item()

        # Normalize labels (0,1)
        self.labels = [label / self.label_scaling_factor for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels, _ = self.data[idx], self.labels[idx], self.lengths[idx]

        # Sort based on the first three features
        sorted_indices = torch.argsort(features[:, 2])
        sorted_indices = sorted_indices[torch.argsort(features[sorted_indices, 1])]
        sorted_indices = sorted_indices[torch.argsort(features[sorted_indices, 0])]
        
        return features[sorted_indices], labels[sorted_indices]
    
def collate_fn(batch):
    features, labels = zip(*batch)
    
    features_padded = pad_sequence([f for f in features], batch_first=True, padding_value=0.0)
    
    # Pad labels based on the maximum length in the batch
    max_label_length = max(len(l) for l in labels)
    labels_padded = pad_sequence(
        [torch.cat([l, torch.full((max_label_length - len(l),), 0)]) for l in labels],
        batch_first=True,
        padding_value=0
    )
    
    return features_padded, labels_padded

def train(encoder, densifier, dataloader, optimizer, criterion, noise, device):
    encoder.train()
    densifier.train()
    total_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        encoded_features = encoder(features)
        
        # Variational Autoencoder to latent -> N(0,1), efficient kl_loss
        kl_loss = 0.5 * encoded_features.pow(2).sum(dim=-1).mean()
        
        encoded_features = encoded_features + torch.randn_like(encoded_features) * noise

        # Densifier step
        densities = densifier(features, encoded_features)
        
        loss = criterion(densities[labels != 0].squeeze(-1), labels[labels != 0])
        
        # Final loss calculation
        loss = loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    total_loss /= len(dataloader)
    
    return total_loss, encoder, densifier


def evaluate(encoder, densifier, dataloader, criterion, device):
    total_loss = 0.0
    print('Testing.')
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            encoded_features = encoder(features)
            densities = densifier(features, encoded_features)
            
            loss = criterion(densities[labels != 0].squeeze(-1), labels[labels != 0])
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def plot_loss(title, train_losses, old_losses=None):
    
    plt.figure(figsize=(10, 6))
    
    if old_losses is not None:
        # Plot old losses in red
        plt.plot(range(len(old_losses)), old_losses, label="Old Training Loss", color="red")
        # Plot new losses in blue
        plt.plot(range(len(old_losses), len(old_losses) + len(train_losses)), 
                 train_losses, label="New Training Loss", color="blue")
    else:
        # If no old losses, plot only the current training loss in blue
        plt.plot(train_losses, label="Training Loss", color="blue")
        
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def show_bone(bone,scale):
    import pyvista as pv
    import numpy as np
    points = []
    cells = []
    
    for elem in bone[0]:
        nodes = elem[:30].reshape(10, 3)  # 10 nodes for each tetrahedron
        points.extend(nodes)
        start_idx = len(points) - 10  
        cells.append([10] + list(range(start_idx, start_idx + 10)))

    points = np.array(points)
    
    cell_type = np.full(len(cells), pv.CellType.TETRA, dtype=np.int8)
    grid = pv.UnstructuredGrid(cells, cell_type, points)
    grid.cell_data['E11'] = bone[1]*scale
    
    plotter = pv.Plotter()
    slices = grid.slice_orthogonal(x=0, y=0, z=0)
    plotter.add_mesh(slices, scalars='E11', show_edges=True, 
                     cmap='viridis',interpolate_before_map=False)
    
    # Create and show random slices
    num_slices = 20 
    for i in range(num_slices):
        x = (np.random.rand(1)-0.5)/100
        y = (np.random.rand(1)-0.5)/100
        z = (np.random.rand(1)-0.5)/100

        random_slice = grid.slice_orthogonal(x=x, y=y, z=z)
        plotter.add_mesh(random_slice, scalars='E11', cmap='viridis',
                         interpolate_before_map=False)
        
    plotter.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=0)
    parser.add_argument('-h1', '--hidden1', type=int, default=8)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('-b', '--bidir', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--pint', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('-v', '--visual', action='store_true')
    
    args = parser.parse_args(['--direct', 'A:/Work/',
                              '--noise','0.0002',
                              '-v',
                              '--batch','64',
                              '-h1','16',
                              '--layers','2',
                              '-lr', '1e-2', '--decay', '1e-6',
                              '-e', '100',
                              '--pint','1',
                              '--optim','adam',
                              '--load', 'med',
                              '--name', 'med'])

    if torch.cuda.is_available():
        print('CUDA available')
        print(torch.cuda.get_device_name(0))
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        print('CUDA *not* available')
        
    # Directories
    train_dir = args.direct + 'Data/inps/Labeled/train.h5'
    test_dir = args.direct + 'Data/inps/Labeled/test.h5'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_epoch = 0
    
    if args.seed == 0:
        args.seed = torch.randint(10, 8545, (1,)).item() 
        
    torch.manual_seed(args.seed)
    print(f'seed: {args.seed}')
    
    # Models
    encoder = dnets.tet10_encoder(args.hidden1, args.layers, args.bidir).to(device)
    densifier = dnets.tet10_densify(args.hidden1).to(device)
    
    # Data Loaders
    train_dataset = MetatarsalDataset(train_dir)
    test_dataset = MetatarsalDataset(test_dir)
    print('Data Loaded.')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
    # Optimizer and Criterion
    param_groups = [
        {'params': encoder.parameters(), 'lr': args.lr * 0.1, 'weight_decay': args.decay},
        {'params': densifier.parameters(), 'lr': args.lr, 'weight_decay': args.decay}
    ]
    
    # Optimizer Dictionary
    optimizer_dict = {
        'adam': lambda: optim.Adam(param_groups),
        'rms': lambda: optim.RMSprop(param_groups, alpha=0.95, eps=1e-8),
        'sgd': lambda: optim.SGD(param_groups, momentum=0.9),
        'adamw': lambda: optim.AdamW(param_groups),
        'adagrad': lambda: optim.Adagrad(param_groups, lr_decay=0)
    }
    
    # Handle optimizer selection
    optimizer_name = args.optim.lower()
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized. Available options are: {list(optimizer_dict.keys())}")
    
    # Instantiate the optimizer
    optimizer = optimizer_dict[optimizer_name]()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    # Criterion
    criterion = nn.HuberLoss()
        
    criterion = nn.HuberLoss()
    train_loss_hist = []
    new_flag = False
    if args.load != '':
        if args.load[-4:] != '.pth': # must be .pth
            args.load += '.pth'
        checkpoint = torch.load(args.direct+'Models/'+args.load)
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f'Successfully loaded {args.load} encoder')
        except:
            print(f'Something went wrong loading {args.load} encoder, starting new!')
            new_flag = True
        
        try:
            densifier.load_state_dict(checkpoint['densifier_state_dict'])
            print(f'Successfully loaded {args.load} densifier')
        except:
            print(f'Something went wrong loading {args.load} densifier, starting new!')
            new_flag = True
        
        if not new_flag:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        train_loss_hist = checkpoint.get('train_losses', [])
    
    # Training Loop
    print('Training.')
    train_losses = []
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss, encoder, densifier = train(
            encoder, densifier, train_loader, optimizer, criterion,
            args.noise, device)
        train_time = time.time() - start 
        train_losses.append(train_loss)
        if epoch % args.pint == 0:
            print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}], '
                  f'Train Time: {train_time:7.2f} s, Train Loss: {train_loss:8.3e}')
        scheduler.step(train_loss)
        
    criterion = nn.L1Loss()
    test_loss = evaluate(encoder, densifier, test_loader, criterion, device)
    
    path = args.direct+'Models/'+args.name+'.pth' 
    
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'densifier_state_dict': densifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_loss_hist+train_losses,
        'testing_loss': test_loss,
        'scale_factor': train_dataset.label_scaling_factor
    }
    torch.save(state, path)
    
    path = f'{args.direct}Metrics/{args.name}_{args.hidden1}x{args.layers}.csv'
    metrics = pd.DataFrame(state['train_losses'])
    metrics.to_csv(path)   
    
    print(f'Saved {args.name}. Test MAE: {test_loss:.3e}')
    
    if args.visual:
        import matplotlib.pyplot as plt
        #print('Showing example input data...')
        #show_bone(test_dataset[0],train_dataset.label_scaling_factor)
        
        if args.load != '':
            plot_loss(args.name,train_losses,train_loss_hist)
        else:
            plot_loss(args.name,train_losses)