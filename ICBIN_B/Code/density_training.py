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
        self.label_scaling_factor = all_labels.max().item() if all_labels.max() > 0 else 1.0

        # Normalize labels
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

def train(encoder, decoder, densifier, dataloader, optimizer, criterion, 
          decode, cycles, loss_mag, device):
    encoder.train()
    densifier.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        indices = features[:,:,-1]
        decoded_features = features.clone()
        
        if(decode):
            decoder.train()
            for i in range(cycles):
                encoded_features = encoder(decoded_features)
                decoded_features = decoder(encoded_features, indices)
                
            loss = criterion(features, decoded_features)
        else:
            loss = 0
            encoder.train()
            densifier.train()
            encoded_features = encoder(decoded_features)
        
        densified_output = densifier(decoded_features, encoded_features)
        
        #Variational Autoencoder to latent -> N(0,1)
        kl_loss = 0.5 * torch.sum(encoded_features.pow(2), dim=-1)
        
        loss += (masked_mse(densified_output.squeeze(-1), labels) + kl_loss.mean())*loss_mag
            
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
    
    import matplotlib.pyplot as plt
    
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
    
def masked_mse(predictions, targets):
    # Remove zeros from MSE measurement
    # Then penalize number of zero predictions
    non_zero_indices = targets != 0
    filtered_predictions = predictions[non_zero_indices]
    filtered_targets = targets[non_zero_indices]
    mse_loss = torch.mean((filtered_predictions - filtered_targets) ** 2)
    
    zero_count = torch.sum(filtered_predictions == 0)
    
    scaling_factor = torch.where(zero_count > 100, 4.0,
                       torch.where(zero_count > 50, 2.0,
                       torch.where(zero_count > 0, 1.5, 1.0)))
    
    return mse_loss * scaling_factor

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
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--pint', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('-a', '--autoencode', action='store_true')
    parser.add_argument('-vae', action='store_true')
    parser.add_argument('-v', '--visual', action='store_true')
    
    args = parser.parse_args(['--direct', 'A:/Work/',
                              '-a',
                              '--cycles','1',
                              '-v',
                              '--batch','64',
                              '-h1','16',
                              '--layers','2',
                              '--experts','2',
                              #'-b',
                              '-lr', '2e-3', '--decay', '1e-4',
                              '-e', '20',
                              '--pint','1',
                              '--loss_mag','1e6',
                              '--optim','adam',
                              '--load', 'lstm_adam',
                              '--name', 'lstm_adam'])

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
    param_groups = list(encoder.parameters()) + list(decoder.parameters()) + list(densifier.parameters())
    
    optimizer_dict = {
        'adam': lambda: optim.Adam(param_groups, lr=args.lr, weight_decay=args.decay),
        'rms': lambda: optim.RMSprop(param_groups, lr=args.lr, alpha=0.95, eps=1e-8, weight_decay=args.decay),
        'sgd': lambda: optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.decay),
        'adamw': lambda: optim.AdamW(param_groups, lr=args.lr, weight_decay=args.decay),
        'adagrad': lambda: optim.Adagrad(param_groups, lr=args.lr, lr_decay=0, weight_decay=args.decay)
    }
    
    optimizer_name = args.optim.lower()
    
    if optimizer_name in optimizer_dict:
        optimizer = optimizer_dict[optimizer_name]()
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Choose from {list(optimizer_dict.keys())}.")
        
    criterion = nn.MSELoss()
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
            
        if args.autoencode:
            try:
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                print(f'Successfully loaded {args.load} decoder')
            except:
                print(f'Something went wrong loading {args.load} decoder, starting new!')
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
    
    print(f'Saved {args.name}.')
    
    if args.visual:
        if args.load != '':
            plot_loss(args.name,train_losses,train_loss_hist)
        else:
            plot_loss(args.name,train_losses)