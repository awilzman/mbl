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
import matplotlib.pyplot as plt
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
        self.label_scaling_factor = all_labels.max().item() if all_labels.max() > 0 else 1.0

        # Normalize labels
        self.labels = [label / self.label_scaling_factor for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels, length = self.data[idx], self.labels[idx], self.lengths[idx]
        perm = torch.randperm(length)
        return features[perm], labels[perm]

    
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

def train(encoder, decoder, densifier, dataloader, optimizer, criterion, 
          decode, cycles, loss_mag, device):
    encoder.train()
    densifier.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        indices = torch.arange(features.size(1), device=device).unsqueeze(0).expand(features.size(0), -1)
        
        if(decode):
            decoder.train()
            decoded_features = features.clone()
            for i in range(cycles):
                encoded_features = encoder(decoded_features)
                decoded_features = decoder(encoded_features, indices)
            loss = criterion(features, decoded_features)
        else:
            encoder.train()
            densifier.train()
            decoded_features = features
            encoded_features = encoder(decoded_features)
        
        densified_output = densifier(decoded_features, encoded_features)
            
        loss += criterion(densified_output.squeeze(-1), labels)*loss_mag
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    total_loss /= len(dataloader)
    
    return total_loss, encoder, decoder, densifier

def evaluate(encoder, decoder, densifier, dataloader, criterion, device):
    total_loss = 0.0
    print('Testing.')
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            encoded_features = encoder(features)
            densified_output = densifier(features, encoded_features)
            
            loss = criterion(densified_output.squeeze(-1), labels)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=0)
    parser.add_argument('-h1', '--hidden1', type=int, default=8)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--experts', type=int, default=1)
    parser.add_argument('-b', '--bidir', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('--loss_mag', type=float, default=1.0)
    parser.add_argument('--decay', type=float, default=1e-6)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--pint', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('-a', '--autoencode', action='store_true')
    parser.add_argument('-vae', action='store_true')
    parser.add_argument('-v', '--visual', action='store_true')
    
    args = parser.parse_args(['--direct', 'A:/Work/',
                              '-a',
                              '--cycles','1',
                              '--experts','4',
                              '-v',
                              '--batch','64',
                              '-h1','16',
                              '--layers','2',
                              '-lr', '1e-3', '--decay', '1e-6',
                              '-e', '30',
                              '--pint','1',
                              '--loss_mag','1e8',
                              '--load', 'lstm',
                              '--name', 'lstm'])

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
    
    # Models
    encoder = dnets.tet10_encoder(args.hidden1, args.layers, args.experts, args.bidir).to(device)
    decoder = dnets.tet10_decoder(args.hidden1).to(device)
    densifier = dnets.tet10_densify(args.hidden1, args.experts).to(device)
        
    # Data Loaders
    train_dataset = MetatarsalDataset(train_dir)
    test_dataset = MetatarsalDataset(test_dir)
    print('Data Loaded.')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
    # Optimizer and Criterion
    optimizer = optim.Adam(list(encoder.parameters()) + 
                           list(decoder.parameters()) + 
                           list(densifier.parameters()), lr=args.lr, weight_decay=args.decay)
        
    criterion = nn.MSELoss()
    train_loss_hist = []
    if args.load != '':
        if args.load[-4:] != '.pth': # must be .pth
            args.load += '.pth'
        checkpoint = torch.load(args.direct+'Models/'+args.load)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if args.autoencode:
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        densifier.load_state_dict(checkpoint['densifier_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss_hist = checkpoint.get('train_losses', [])
    
    # Training Loop
    print('Training.')
    train_losses = []
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss, encoder, decoder, densifier = train(
            encoder, decoder, densifier, train_loader, optimizer, criterion,
            args.autoencode, args.cycles, args.loss_mag, device)
        train_time = time.time() - start 
        train_losses.append(train_loss)
        if epoch % args.pint == 0:
            print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}], '
                  f'Train Time: {train_time:7.2f} s, Train Loss: {train_loss:8.3e}')
    
    criterion = nn.L1Loss()
    test_loss = evaluate(encoder, decoder, densifier, test_loader, criterion, device)
    
    path = args.direct+'Models/'+args.name+'.pth' 
    
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
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
    
    print('Saved.')
    
    if args.visual:
        if args.load != '':
            plot_loss(args.name,train_losses,train_loss_hist)
        else:
            plot_loss(args.name,train_losses)