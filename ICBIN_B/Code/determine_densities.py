# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:43:18 2024

@author: arwilzman

v1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import h5py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import time
import matplotlib.pyplot as plt

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
        return features[perm], labels[perm], length

    
def collate_fn(batch):
    features, labels, lengths = zip(*batch)
    
    # Pad sequences of features to the same length
    features_padded = pad_sequence([f for f in features], batch_first=True, padding_value=0.0)
    
    # Pad labels based on the maximum length in the batch
    max_label_length = max(len(l) for l in labels)
    labels_padded = pad_sequence(
        [torch.cat([l, torch.full((max_label_length - len(l),), 0)]) for l in labels],
        batch_first=True,
        padding_value=0
    )
    
    lengths = torch.tensor(lengths, dtype=torch.int64)
    
    return features_padded, labels_padded, lengths

def train(encoder, decoder, densifier, dataloader, optimizer, criterion, decode, device):
    encoder.train()
    decoder.train()
    densifier.train()
    total_loss = 0.0

    for features, labels, lengths in dataloader:
        features, labels = features.to(device), labels.to(device)
        lengths = lengths.detach().cpu()

        encoded_features = encoder(features, lengths)

        # Create a tensor of indices for the sequence length
        indices = torch.arange(features.size(1), device=device).unsqueeze(0).expand(features.size(0), -1)
        
        # Use batch processing instead of a loop
        if(decode):
            decoded_features = decoder(encoded_features, indices)
        else:
            decoded_features = features
        
        densified_output = densifier(decoded_features, encoded_features)
        
        loss = criterion(densified_output.squeeze(-1), labels)/lengths.mean(dtype=float)
        
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
        for features, labels, lengths in dataloader:
            features, labels = features.to(device), labels.to(device)
            lengths = lengths.detach().cpu()
            
            encoded_features = encoder(features, lengths)
            densified_output = densifier(features, encoded_features)
            
            loss = criterion(densified_output.squeeze(-1), labels)/lengths.mean(dtype=float)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

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
        self.dropout = nn.Dropout(0.1)
        # Output size depends on whether the GRU is bidirectional
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, lengths):
        # Pack padded sequences for variable-length inputs
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # Unpack the sequence

        last_hidden_states = output[torch.arange(output.size(0)), lengths - 1, :]
        encoded = self.fc1(last_hidden_states)
        
        encoded = self.fc2(torch.relu(self.dropout(encoded)))
        encoded = self.fc3(torch.relu(self.dropout(encoded)))

        return encoded
    
class tet10_decoder(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_decoder, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        self.fc1 = nn.Linear(self.codeword_size + 1, 16)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(16, self.feature_size)
        
    def forward(self, x, i):
        x = x.unsqueeze(1)  # Shape: [B, 1, D]
        i = i.unsqueeze(-1)  # Shape: [B, E, 1]
        
        x = x.expand(-1, i.size(1), -1)  # Shape: [B, E, D]
        
        x = torch.cat((x, i), dim=-1)  # Shape: [B, E, D + 1]
        
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
    
class tet10_densify(nn.Module):
    def __init__(self, codeword_size=16):
        super(tet10_densify, self).__init__()
        self.codeword_size = codeword_size
        self.feature_size = 30
        
        # Define the network layers
        self.fc1 = nn.Linear(self.feature_size + self.codeword_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x, encoded_features):
        # x shape: (B, E, 30) - Original features
        # encoded_features shape: (B, 1, 16) - Encoded features
        B, E, _ = x.size()
        
        encoded_features_repeated = encoded_features.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, E, 16)
        
        combined = torch.cat((x, encoded_features_repeated), dim=2)  # (B, E, 30 + 16)
        
        x = self.fc1(combined)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        
        return x

def plot_loss(train_losses, old_losses=None):
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
    plt.title("Training Loss Over Epochs")
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
    parser.add_argument('-b', '--bidir', action='store_true')
    parser.add_argument('-t', '--traintime', type=int, default=600)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1e-6)
    parser.add_argument('--chpt', type=int, default=0)
    parser.add_argument('--chkdecay', type=float, default=0.95)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--pint', type=int, default=1)
    parser.add_argument('--noise', type=int, default=3)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('-a', '--autoencode', action='store_true')
    parser.add_argument('-vae', action='store_true')
    parser.add_argument('-v', '--visual', action='store_true')
    
    args = parser.parse_args(['--direct', 'A:/Work/',
                              '-a',
                              '-v',
                              #'-b',
                              '-lr', '1e-3', '--decay', '1e-5',
                              '-e', '30',
                              #'--load', 'det_den_models2',
                              '--name', 'det_den_models1'])

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
    encoder = tet10_encoder(args.hidden1, args.layers, args.bidir).to(device)
    decoder = tet10_decoder(args.hidden1).to(device)
    densifier = tet10_densify(args.hidden1).to(device)
        
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
            encoder, decoder, densifier, train_loader, optimizer, criterion,args.autoencode, device)
        train_time = time.time() - start 
        train_losses.append(train_loss)
        print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}], '
              f'Train Time: {train_time:7.2f} s, Train Loss: {train_loss:8.3e}')
    
    test_loss = evaluate(encoder, decoder, densifier, test_loader, criterion, device)
    
    path = args.direct+'Models/'+args.name+'.pth' 
    
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'densifier_state_dict': densifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_loss_hist+train_losses
    }
    torch.save(state, path)
    print('Saved.')
    
    if args.visual:
        if args.load != '':
            plot_loss(train_losses,train_loss_hist)
        else:
            plot_loss(train_losses)