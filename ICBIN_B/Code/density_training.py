# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:46:23 2024

@author: Andrew
"""
import torch
import h5py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
        
        if(decode):
            loss += criterion(features,decoded_features)/lengths.mean(dtype=float)
            
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