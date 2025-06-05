# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:46:23 2024
Updated 04/13/25
@author: Andrew
v5.0
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import tabulate
import argparse
import time
import pandas as pd
import density_networks as dnets
import os
import numpy as np
import re

class MetatarsalDataset(Dataset):
    def __init__(self, folder, output_size, label_scaling_factor=None):
        self.folder = folder
        self.output_size = output_size
        self.files = [f for f in os.listdir(folder) if f.endswith("_raw.npy")]
        
        if 'Runner' in folder:
            self.runner=True
        else:
            self.runner=False
            
        self.parent_map = self._build_parent_map()
        
        if label_scaling_factor is None:
            # Precompute label scaling factor using NumPy for better efficiency
            max_vals = np.full(self.output_size, -np.inf)
            for f in self.files[:100]:
                data = np.load(os.path.join(folder, f), mmap_mode='r').astype("float32")
                if data.shape[1] == self.output_size:
                    lbl = data[1:, :]
                else:
                    lbl = data[1:, 31:31+output_size]

                max_vals = np.maximum(max_vals, np.abs(lbl).max(axis=0))

            self.label_scaling_factor = torch.from_numpy(max_vals.astype("float32"))
        else:
            self.label_scaling_factor = label_scaling_factor.clone()
        print(self.label_scaling_factor)
        print('Loaded all data!')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load & convert once
        arr = np.load(os.path.join(self.folder, self.files[idx]), mmap_mode="r").astype("float32")
        data = torch.from_numpy(arr)            # <-- single conversion
        h = torch.Tensor(data[0:1, :7]) # 7 meta cols see inp_sleth.py
        # load (N), angle (deg), age (yr), height (cm), weight (kg), sex (M=-1, F=1), runner (Y=1,N=0,C=-1) C : Cadaver
        if data.size(1) == self.output_size:
            lbl = data[1:, :]
            pd = torch.from_numpy(
                np.load(os.path.join(self.folder, self.parent_map[self.files[idx]]),
                        mmap_mode="r").astype("float32"))
            x = pd[1:, :30]
            s = pd[1:, 30:31]
        else:
            x = data[1:, :30]
            s = data[1:, 30:31]
            lbl = data[1:, 31:31 + self.output_size]

        if self.label_scaling_factor is not None:
            lbl = lbl / self.label_scaling_factor

        try:
            inp = torch.cat((x, s, lbl), dim=1) #label is diffused out during training
        except:    
            print(self.files[idx])
            print(self.parent_map[self.files[idx]])
        return inp, h, lbl

    def _build_parent_map(self):
        """ Precompute mapping from augmented file to parent file """
        mapping = {}
    
        def normalize(fname):
            # Standardize study ID to MTSFX_02 format
            return re.sub(r'^([A-Z]*)?(\d{2})(_.+)', r'\1\2\3', fname)
    
        for f in self.files:
            norm_f = normalize(f)
            base = norm_f.split("_")
    
            key = '_'.join(base[0:4])
    
            matches = []
            for x in self.files:
                norm_x = normalize(x)
                if key in norm_x:
                    matches.append(x)
            parent = min(matches, key=len) if matches else None
            mapping[f] = parent
    
        return mapping

def collate_fn(batch):
    x, h, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)
    y_pad = pad_sequence(y, batch_first=True, padding_value=0)
    return x_pad, torch.stack(h), y_pad
    
def train(model, scheduler, 
          dataloader, start_epoch, epochs, 
          optimizer, criterion, noise, pint, device):
    model.train()
    # Track losses
    supervised_losses = []
    
    if start_epoch >= epochs:
        epochs += start_epoch
    
    gradual = False
        
    # Training Loop
    print('Training.')
    train_losses = []
    for epoch in range(start_epoch, epochs):
        start = time.time()
        if gradual:
            progress = (epoch - start_epoch) / (epochs - start_epoch)
            current_noise = noise * progress
        else:
            current_noise = noise
            
        for features, headers, labels in dataloader:
            out = features.shape[2]-31
            features, headers, labels = features.to(device), headers.to(device), labels.to(device)
            optimizer.zero_grad()
            
            mask = (torch.rand(features.size(0), features.size(1), 1, 
                               device=features.device) > current_noise).float()
            features[:, :, -out:] *= mask
            
            # Encoder step
            mod = model(features,headers)

            # Feature-wise normalization
            label_splits = torch.split(labels, 1, dim=2)
            mod_splits = torch.split(mod, 1, dim=2)

            w_mods = []
            w_labels = []

            weights = []

            for m, lab in zip(mod_splits, label_splits):
                m = m.squeeze(2)
                lab = lab.squeeze(2)
                mask = lab != 0
                n = mask.sum().item()
                if n > 0:
                    w_mods.append(m[mask])
                    w_labels.append(lab[mask])
                    weights.append(1.0 / n)

            # Normalize weights to sum to 1
            weights = torch.tensor(weights, device=labels.device)
            weights = weights / weights.sum()

            # Concatenate
            mod_all = torch.cat(w_mods)
            labels_all = torch.cat(w_labels)
            
            supervised_loss = criterion(mod_all, labels_all)
            
            supervised_loss.backward()
            optimizer.step()
        
        train_time = time.time() - start
        sup = float(supervised_loss)
        supervised_losses.append(sup)
        if epoch % pint == 0:
            print(f'Epoch [{epoch+1:4d}/{epochs:4d}], '
                  f'Train Time: {train_time:7.2f} s, Train Loss: {sup:8.3e}')
        
        scheduler.step(supervised_loss)
    
    return supervised_losses, model

def evaluate(model, dataloader, criterion, num_out, device):
    total_loss = 0.0
    print('Testing.')
    model.eval()
    with torch.no_grad():
        for features, headers, labels in dataloader:
            features, headers, labels = features.to(device), headers.to(device), labels.to(device)
            features[:, :, -num_out:] = 0 #remove labels for testing
            mod = model(features,headers)
            
            mask = labels != 0
            mod = mod[mask]
            labels = labels[mask]
            
            loss = criterion(mod, labels)
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

def show_bone(bone, scale, title='', scales_in=None):
    import pyvista as pv
    points = []
    cells = []

    for i, elem in enumerate(bone[0]):
        nodes = elem[:30].reshape(10, 3)
        points.extend(nodes)
        start_idx = len(points) - 10
        cells.append([10] + list(range(start_idx, start_idx + 10)))

    points = np.array(points)
    cell_type = np.full(len(cells), pv.CellType.TETRA, dtype=np.int8)
    grid = pv.UnstructuredGrid(cells, cell_type, points)

    values = (bone[2] * scale).T  # shape: (output_size, n_cells)
    scalars = ['E11', 'Max Pcpl Strain', 'Min Pcpl Strain', 'TW Strain',
               'vM Strain', 'Ele Vol', 'vM Stress']
    scalars = [title + ' ' + i for i in scalars]

    reported_scales = []

    for i, scalar_values in enumerate(values):
        grid.cell_data.clear()
        grid.cell_data[scalars[i]] = scalar_values

        # Determine scale range
        if scales_in and i < len(scales_in) and scales_in[i] is not None:
            vmin, vmax = scales_in[i]
        else:
            vmin = float(scalar_values.min())
            vmax = float(scalar_values.max())

        reported_scales.append((vmin, vmax))

        plotter = pv.Plotter()
        slices = grid.slice_orthogonal(x=0, y=0, z=0)
        plotter.add_mesh(slices, scalars=scalars[i], show_edges=True,
                         cmap='viridis', interpolate_before_map=False,
                         clim=[vmin, vmax])
        plotter.add_text(scalars[i], font_size=12)

        for _ in range(2):
            x = (np.random.rand(1) - 0.5) / 100
            y = (np.random.rand(1) - 0.5) / 100
            z = (np.random.rand(1) - 0.5) / 100
            s = grid.slice_orthogonal(x=x, y=y, z=z)
            plotter.add_mesh(s, scalars=scalars[i], cmap='viridis',
                             interpolate_before_map=False,
                             clim=[vmin, vmax], smooth_shading=False)

        plotter.show()

    return reported_scales

def encode_and_save(input_folder, output_name, model, device):
    
    # read raw element data and save embedded versions!!

    for filename in os.listdir(input_folder):
        if filename.endswith("_raw.npy"):
            file_path = os.path.join(input_folder, filename)
            out_file = filename.replace('_raw',f'_{output_name}')
            output_path = os.path.join(input_folder, out_file)
            data = np.load(file_path).astype("float32")
            data_tensor = torch.from_numpy(data).float().to(device)

            # need to apply scaling factors and noise and header            
            embedded_data = model.encode(data_tensor.unsqueeze(0))
            
            embedded_data = embedded_data.detach().cpu().numpy()

            np.save(output_path, embedded_data.squeeze(2))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=0)
    parser.add_argument('-h1', '--hidden1', type=int, default=32)
    parser.add_argument('-o', '--output_size', type=int, default=1)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('-b', '--bidir', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-a', '--aug', action='store_true')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--pint', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('-ld', '--loaddis', action='store_true')
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('-v', '--visual', action='store_true')
    parser.add_argument('-sv', '--save', action='store_true')
    parser.add_argument('-r', '--runner', action='store_true')
    args = parser.parse_args(['--direct','../',
                              '-h1','16',
                              '-o','7',
                              '-e','1',
                              '--batch','128',
                              #'-v',
                              #'-r',
                              '--noise','0',
                              '--name','test',
                              '--load',''])

    
    torch.cuda.empty_cache()
    torch.cuda.init()
    # Directories
    if args.runner:
        train_dir = args.direct + 'Data/Fatigue/FE_Runner/'
    else:
        train_dir = args.direct + 'Data/Fatigue/FE_Cadaver/'
        
    test_dir = args.direct + 'Data/Fatigue/FE_Cadaver/'
    
    start_epoch = 0
    
    if args.seed == 0:
        args.seed = torch.randint(10, 8545, (1,)).item() 
        
    torch.manual_seed(args.seed)
    print(f'seed: {args.seed}')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
        print(torch.cuda.get_device_name(0))
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        device = torch.device('cpu')
        print('CUDA *not* available')
        
    # Models
    output_size = args.output_size
    
    model = dnets.tet10_autoencoder(args.hidden1, output_size).to(device)
    #discriminator = dnets.tet10_discriminator(args.hidden1).to(device)
    
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #dis_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f'Parameter count: {no_params}')
    
    # Optimizer and Criterion
    params = [
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.decay}
    ]
    
    # Optimizer Dictionary
    optimizer_dict = {
        'adam': lambda: optim.Adam(params),
        'rms': lambda: optim.RMSprop(params, alpha=0.95, eps=1e-8),
        'sgd': lambda: optim.SGD(params, momentum=0.9),
        'adamw': lambda: optim.AdamW(params),
        'adagrad': lambda: optim.Adagrad(params, lr_decay=0)
    }
    # Instantiate the optimizers
    optimizer = optimizer_dict[args.optim]()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
       
    train_loss_hist = []
    label_scaling_factor = None
    
    if args.load != '':
        if args.load[-4:] != '.pth': # must be .pth
            args.load += '.pth'
        checkpoint = torch.load(args.direct+'Models/'+args.load, weights_only=False)
        label_scaling_factor = checkpoint['scale_factor']
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print(f'Successfully loaded {args.load}')
        except:
            print(f'Something went wrong loading {args.load}, starting new!')
        
        start_epoch = checkpoint['epoch']
        
        train_loss_hist = checkpoint.get('train_losses', [])
    
    # Data Loader
    train_dataset = MetatarsalDataset(train_dir, output_size, label_scaling_factor)
    if test_dir != train_dir:
        test_dataset = MetatarsalDataset(test_dir,output_size,train_dataset.label_scaling_factor)
    else:
        test_dataset = train_dataset
    
    criterion = nn.L1Loss()
    
    if args.visual:
        import matplotlib.pyplot as plt
        print('Showing example input data...')
        X = train_dataset[1]
        scales = show_bone(X,train_dataset.label_scaling_factor,'Real')
        
        X[0][:, -output_size:] = 0
        model.eval()
        with torch.no_grad():
            output = model(X[0].unsqueeze(0).to(device),X[1].unsqueeze(0).to(device))
        X[2][:] = output.squeeze(0)
        
        _ = show_bone(X,train_dataset.label_scaling_factor,'Reconstructed',scales)
        
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              shuffle=True, collate_fn=collate_fn)
    
    train_loss, model = train(model, scheduler, train_loader, start_epoch, args.epochs, 
        optimizer, criterion, args.noise, args.pint, device)
    
    # Evaluation
    train_scale = train_dataset.label_scaling_factor
    
    state = {
        'state_dict': model.state_dict(),
        'epoch': args.epochs,
        'train_losses': train_loss_hist + train_loss,
        'scale_factor': train_dataset.label_scaling_factor
    }
    path = args.direct+'Models/'+args.name+'.pth' 
    torch.save(state, path)
    
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    criterion = nn.L1Loss()

    test_loss = evaluate(model, test_loader, criterion, output_size, device)
    
    path = f'{args.direct}Metrics/{args.name}_{args.hidden1}.csv'
    metrics = pd.DataFrame(state['train_losses'])
    metrics.to_csv(path)
    
    if args.save:
        import os
        import numpy as np
        
        path2 = args.direct+'Data/Fatigue/Models/'+args.name+'_enc.pth' 
        torch.save(state, path2)
        
        encode_and_save('Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Fatigue',
                        args.name,model,device)
        
        encode_and_save('Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Fatigue/Test_Data',
                        args.name,model,device)
        
        encode_and_save('Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Fatigue/Runner',
                        args.name,model,device)
    
    
    print(f'Saved {args.name}. Test MAE: {test_loss:.3e}')
    
    
