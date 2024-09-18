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
import time
import pandas as pd
import density_networks as dnets
import density_training as dtrn

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
                              '-h1','64',
                              '--layers','1',
                              '-lr', '1e-2', '--decay', '1e-5',
                              '-e', '60',
                              '--load', 'det_den_models4',
                              '--name', 'det_den_models4'])

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
    encoder = dnets.tet10_encoder(args.hidden1, args.layers, args.bidir).to(device)
    decoder = dnets.tet10_decoder(args.hidden1).to(device)
    densifier = dnets.tet10_densify(args.hidden1).to(device)
        
    # Data Loaders
    train_dataset = dtrn.MetatarsalDataset(train_dir)
    test_dataset = dtrn.MetatarsalDataset(test_dir)
    print('Data Loaded.')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=dtrn.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=dtrn.collate_fn)
        
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
        train_loss, encoder, decoder, densifier = dtrn.train(
            encoder, decoder, densifier, train_loader, optimizer, criterion,args.autoencode, device)
        train_time = time.time() - start 
        train_losses.append(train_loss)
        print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}], '
              f'Train Time: {train_time:7.2f} s, Train Loss: {train_loss:8.3e}')
    
    test_loss = dtrn.evaluate(encoder, decoder, densifier, test_loader, criterion, device)
    
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
    
    path = args.direct+'Metrics/'+args.name+'.csv'
    metrics = pd.DataFrame(state['train_losses'])
    metrics.to_csv(path)   
    
    print('Saved.')
    
    if args.visual:
        if args.load != '':
            dtrn.plot_loss(args.name,train_losses,train_loss_hist)
        else:
            dtrn.plot_loss(args.name,train_losses)