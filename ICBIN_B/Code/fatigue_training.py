# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 08:49:01 2025

@author: arwilzman
"""
import os
import re
import openpyxl
import glob
import h5py
import time
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import density_networks as dnets
import networks as snets
from sklearn.model_selection import KFold

class FailurePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, shape_hidden=512, seq_len=600):
        super().__init__()

        self.h = 512
        self.seq_len = seq_len
        self.act = nn.LeakyReLU()

        self.input_proj = nn.Linear(input_dim, self.h)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.h))

        self.q_proj = nn.Linear(self.h, self.h)
        self.k_proj = nn.Linear(self.h, self.h)
        self.v_proj = nn.Linear(self.h, self.h)
        self.attn_pool = nn.Linear(self.h, 1)

        self.proj_encoding = nn.Linear(hidden_dim, self.h)
        self.proj_shape = nn.Linear(shape_hidden, self.h)

        self.fc1 = nn.Linear(self.h, self.h // 4)
        self.fc2 = nn.Linear(self.h // 4, self.h // 8)
        self.fc2_5 = nn.Linear(4, 1)
        self.fc3 = nn.Linear(self.h // 8, self.h // 16)
        self.fc4 = nn.Linear(self.h // 16, 4)
        self.fc_out = nn.Linear(4, 1)

        self.dropout = nn.Dropout(0.05)

    def forward(self, X, enc, shape):
        B, L, _ = X.shape
        cycles = X[:, :, 0].unsqueeze(2) + 1

        # Positional encoding
        x = self.input_proj(X) + self.pos_embed[:, :L]

        # Attention mechanism
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scale = torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32, device=Q.device))
        scores = Q @ K.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ V

        pool_weights = F.softmax(self.attn_pool(attn_out), dim=1)
        x = (attn_out * pool_weights).sum(dim=1, keepdim=True)  # shape (B, 1, D)

        # Process additional inputs
        enc_proj = self.act(self.proj_encoding(enc)).unsqueeze(1)
        shape_proj = self.act(self.proj_shape(shape))

        x = torch.cat((x, enc_proj, shape_proj), dim=1)

        x = self.act(self.dropout(self.fc1(x)))
        x = self.act(self.dropout(self.fc2(x)))
        x = self.act(self.fc2_5(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.act(self.dropout(self.fc3(x)))
        x = self.act(self.fc4(x))
        out = self.act(self.fc_out(x))
        actual = self.act(out)

        # Weibull failure probability
        k = 2.0
        log2 = torch.log(torch.tensor(2.0, dtype=torch.float32, device=x.device))
        λ = 250000 * actual.expand(-1, L, -1) / (log2 ** (1.0 / k))
        output = 1 - torch.exp(-((cycles / λ) ** k))

        return output, actual

class FailDataset(Dataset):
    def __init__(self, device, folder_path, load_cases, max_seq_len=600,running=False,samp_mult=1):
        self.device = device
        self.max_seq_len = max_seq_len  
        self.LC = load_cases
        self.samples = samp_mult
        self.running = running
        
        if running:
            self.folder_path = os.path.join(folder_path, 'FE_Runner')
            self.files = [f for f in os.listdir(self.folder_path) if f.endswith("30deg_raw.npy")]
        else:
            self.folder_path = os.path.join(folder_path, 'Mech_Test')
            self.files = [f for f in os.listdir(self.folder_path) if f.endswith(".parquet")]

    def __len__(self):
        return len(self.files) * self.samples

    def __getitem__(self, idx):
        idx = idx // self.samples
        file_path = os.path.join(self.folder_path, self.files[idx])
                    
        segm = self.files[idx].split('_')
        if 'R15BSI' in self.files[idx]:
            segm[0] += '_' + segm[1]
            segm[1] = segm[2]
            segm[2] = segm[3]
        
        load = self.LC[
            (self.LC['Foot'] == segm[0]) & 
            (self.LC['Side'] == segm[1]) & 
            (self.LC['Mtno'] == int(segm[2][0]))
        ]

        if load.empty:
            raise ValueError(f"Load case not found for {self.files[idx]}")
            
        shape_file = os.path.join(self.folder_path, f'../../Compressed/{segm[0]}/{segm[1]}{int(segm[2][0])}.h5')
        
        try:
            with h5py.File(shape_file, 'r') as hf:
                bone = hf['Surface'][:]
        except Exception as e:
            print(f"Error loading surface file: {shape_file}, {e}")
            bone = np.zeros((1024, 3), dtype=np.float32)
            
        num_samples = min(1024, bone.shape[0])
        sampled_indices = torch.randperm(bone.shape[0])[:num_samples]
        point_cloud = torch.FloatTensor(bone[sampled_indices]).to(self.device)
        
        if self.running: #fill with 0s except cycles
            cols = ['Cycle', 'Loading Stiffness', 'Unloading Stiffness', 'Energy Dissipation', 'Failure Probability']
            df = pd.DataFrame(np.zeros((self.max_seq_len, len(cols))), columns=cols)
            df['Cycle'] = np.logspace(np.log10(1), np.log10(250000), num=self.max_seq_len)
        else:
            df = pd.read_parquet(file_path)
            if df.shape[0] == 0:
                print(f'Empty file: {file_path}')
        
        X = df.iloc[:, :-1].values.astype('float32')
        y = df.iloc[:, -1].values.astype('float32')
        X, y = self._pad_or_truncate(X, y, self.max_seq_len)

        noise1 = torch.rand(X.shape[0]).to(self.device).unsqueeze(1)
        noise2 = (torch.rand(X.shape[0]).to(self.device) * 0.5).unsqueeze(1)

        load_ = torch.FloatTensor(load['Load'].values).repeat(self.max_seq_len, 1).to(self.device)
        angle_ = torch.FloatTensor(load['Angle'].values).repeat(self.max_seq_len, 1).to(self.device)

        load_ += noise1
        angle_ += noise2

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        X_tensor = torch.cat([X_tensor, load_, angle_], dim=1)

        if self.running:
            pattern = os.path.join(self.folder_path, self.files[idx])
            encodings = [pattern]
        else:
            filename = f"{segm[0]}_{segm[1]}_MT{segm[2][0]}_*{load['Angle'].values[0]}deg_*raw.npy"
                
            folder = os.path.join(self.folder_path, '../FE_Cadaver')
            encodings = glob.glob(os.path.join(folder, filename))
            

        if not encodings:
            raise FileNotFoundError(f"No matching FE files found for pattern: {filename}")
        
        match_found = False  # Flag to track if a valid file is found

        for j in encodings:
            # Skip files that contain '_p<digit>_' or '_n<digit>_'
            if '_p' in j or '_n' in j:
                continue
        
            # Now check for the Load pattern (_<digit>N) in the filename
            match = re.search(r'_(\d+)N', j)
            
            if match:
                # Check if the Load value matches the current load
                load_value = int(match.group(1))
                if load_value == int(load['Load'].values[0]):
                    elements = torch.tensor(np.load(j).astype("float32"), dtype=torch.float32).to(self.device)
                    match_found = True
                    break  # Exit the loop once a valid file is found
        
        # If no valid file was found
        if not match_found:
            print("Woah there, no file matches the pattern.")
        
        fatigue_life = torch.tensor(load.iloc[0]['Fatigue Life'], dtype=torch.float32).to(self.device)

        return X_tensor, elements, point_cloud, y_tensor, fatigue_life

    def _pad_or_truncate(self, X, y, max_len):
        seq_len = X.shape[0]
        
        if seq_len > max_len:
            X = X[:max_len, :]
            y = y[:max_len]
        elif seq_len < max_len:
            pad_X = np.full((max_len, X.shape[1]), -1, dtype=np.float32)
            pad_y = np.full((max_len,), -1, dtype=np.float32)
            pad_X[:seq_len, :] = X
            pad_y[:seq_len] = y

            valid_indices = np.where(X[:, 0] != -1)[0]
            start_value = X[valid_indices[-1], 0] if len(valid_indices) > 0 else 1
            lin_values = np.linspace(start_value, 250000, num=max_len - seq_len)

            pad_X[seq_len:, 0] = lin_values
            pad_X[seq_len:, -2:] = X[-1, -2:]

            X, y = pad_X, pad_y

        return X, y

def collate_fn(batch):
    X_list, elem_list, pc_list, y_list, fl_list = zip(*batch)
    X_batch = torch.stack(X_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    fl_batch = torch.stack(fl_list, dim=0)
    pc_batch = torch.stack(pc_list, dim=0)
    elem_batch = pad_sequence(elem_list, batch_first=True, padding_value=-1)
    return X_batch, elem_batch, pc_batch, y_batch, fl_batch

def split_dataset(dataset, test_ratio=0.2):
    
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    
    return random_split(dataset, [train_size, test_size])

def train(model, s_enc, d_enc, train_loader, criterion, m_opt, d_opt, s_opt, 
          noise, m_sch, s_sch, d_sch, device, pint, start_e=0, num_epochs=10):
    
    losses = []
    start_time = time.time()
    for epoch in range(start_e, num_epochs):
        model.train()
        s_enc.train()
        d_enc.train()
        total_loss = 0.0

        for X_tensor, element_data, point_cloud, y_tensor, fatigue_life in train_loader:
            X_tensor, element_data, point_cloud, y_tensor, fatigue_life = (
                X_tensor.to(device), element_data.to(device), point_cloud.to(device), 
                y_tensor.to(device), fatigue_life.to(device))
            
            # Diffuse on mechanical test outputs (Loading K, unloading K, energy dissipation)
            mask = (torch.rand(X_tensor.size(0), X_tensor.size(1), 3, device=device) > noise)
            X_tensor[:, :, 1:4][mask] = 0
            
            # encode shape and density
            
            shape_enc,_ = s_enc.encode(point_cloud)
            
            encoding_tensor = d_enc.encode(element_data)
            
            # Forward pass
            pred, actual = model(X_tensor, encoding_tensor, shape_enc)
            pred = pred.squeeze(-1)
            actual = actual.squeeze(-1).squeeze(-1)*250000
            
            loss = criterion(pred[y_tensor>0], y_tensor[y_tensor>0])
            anneal = min(1.0, epoch / 20)
            loss += anneal*F.l1_loss(actual, fatigue_life, reduction="mean")/1e6
            
            std_actual = torch.std(actual)
            std_fatigue = torch.std(fatigue_life)
            loss += anneal*F.mse_loss(std_actual, std_fatigue, reduction="mean")
            
            m_opt.zero_grad()
            s_opt.zero_grad()
            d_opt.zero_grad()
            loss.backward()
            m_opt.step()
            #s_opt.step()
            #d_opt.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        m_sch.step(avg_loss)
        s_sch.step(avg_loss)
        d_sch.step(avg_loss)
        
        if epoch % pint == 0:
            ep_time = time.time() - start_time
            print(f"Epoch [{epoch+1} / {num_epochs}], Loss: {avg_loss:.3e}, Time: {ep_time:.1f}s")

    return model, s_enc, d_enc, losses

def find_and_match_ids(data_paths, excel_path):

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    excel_ids = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        val = str(row[0]).strip() if row[0] else ''
        if val:
            excel_ids.add(val)

    matches = []
    pattern = re.compile(r'([A-Z]+\d+)')

    for path in data_paths:
        base = os.path.basename(path)
        match = pattern.search(base)
        key = match.group(1) if match else ''
        is_valid = key in excel_ids
        matches.append((path, base, os.path.splitext(base)[0], key, is_valid))

    return matches

def evaluate(model, s_enc, d_enc, test_loader, criterion, device):
    model.eval()
    s_enc.eval()
    d_enc.eval()
    
    total_loss = 0.0
    w_predictions = []
    w_targets = []
    predictions = []
    targets = []

    with torch.no_grad():
        for X_tensor, element_data, point_cloud, y_tensor, fatigue_life in test_loader:
            X_tensor = X_tensor.to(device)
            element_data = element_data.to(device)
            point_cloud = point_cloud.to(device)
            y_tensor = y_tensor.to(device)
            fatigue_life = fatigue_life.to(device)

            X_tensor[:, :, 1:4] *= 0  # zero out columns 1–3

            shape_enc, _ = s_enc.encode(point_cloud)
            encoding_tensor = d_enc.encode(element_data)

            pred, pred_f = model(X_tensor, encoding_tensor, shape_enc)

            loss = criterion(pred.squeeze(2)[y_tensor != -1], y_tensor[y_tensor != -1])
            total_loss += loss.item()

            w_predictions.append(pred.cpu().numpy())
            w_targets.append(y_tensor.cpu().numpy())

            predictions.append(pred_f.cpu().numpy() * 250000)
            targets.append(fatigue_life.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    predictions = np.concatenate(predictions, axis=0).squeeze(1).squeeze(1)
    targets = np.concatenate(targets, axis=0)
    w_predictions = np.concatenate(w_predictions, axis=0)
    w_targets = np.concatenate(w_targets, axis=0)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print(f"Test Loss: {avg_loss:.3e}")
    print(f"Fatigue Life Mean Absolute Error: {mae:.3e}")
    print(f"Fatigue Life R2: {r2:.3f}")

    return avg_loss, predictions, targets, w_predictions, w_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=0)
    parser.add_argument('-h1', '--hidden1', type=int, default=32)
    parser.add_argument('-se1', type=int, default=512)
    parser.add_argument('-seq', type=int, default=600) 
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('-a','--augment', type=int, default=1)
    parser.add_argument('--pint', type=int, default=1)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('-s','--shape_enc_name', type=str, default='Vae_tin3_512_128_128_0')
    parser.add_argument('-d','--dense_enc_name', type=str, default='real2')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('-c', '--continue_encoders', action='store_true')
    parser.add_argument('-v', '--visual', action='store_true')
    parser.add_argument('-LP', '--lots_plots', action='store_true')
    parser.add_argument('-SP', '--save_preds', action='store_true')
    args = parser.parse_args(['--direct','../Data/Fatigue/',
                              '-v',
                              '-h1','512',
                              '--name','test',
                              '--load','test',#'-c',
                              '-e','10'])
    
    se_hidden2 = 128
    dnet_outsize = 7
    
    torch.cuda.empty_cache()
    torch.cuda.init()
    
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
    
    # Load encoders
    if args.continue_encoders:
        s_check = torch.load(args.direct+'Models/'+args.load+'_shape_encoder.pth', weights_only=False)
        d_check = torch.load(args.direct+'Models/'+args.load+'_density_encoder.pth', weights_only=False)
        print('continuing encoders')
    else:
        s_check = torch.load(args.direct+'../../Models/'+args.shape_enc_name+'.pth', weights_only=False)
        d_check = torch.load(args.direct+'../../Models/'+args.dense_enc_name+'.pth', weights_only=False)
    
    s_model = snets.arw_FoldingNet(args.se1,se_hidden2).to(device)
    s_model.load_state_dict(s_check)
    
    d_model = dnets.tet10_autoencoder(args.hidden1,dnet_outsize).to(device)
    d_model.load_state_dict(d_check['state_dict'])
    
    load_cases = pd.read_excel(f'{args.direct}/Cadaver_Loadcases.xlsx')
    
    seq_len = 600
    dataset = FailDataset(device, args.direct, load_cases, seq_len,False,args.augment)
    
    ## I need to fix how I do this
    #validation_set = FailDataset(device, args.direct+'Test_Data/', load_cases, seq_len)
    
    train_dataset, test_dataset = split_dataset(dataset)
    
    print(f'training length: {len(train_dataset)}, testing length: {len(test_dataset)}')
    
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    #valid_loader = DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    #%%
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    fp_model = FailurePredictor(6,args.hidden1,args.se1,seq_len).to(device)
    
    m_params = sum(p.numel() for p in fp_model.parameters() if p.requires_grad)
    print(f'Failure Predictor parameter count: {m_params}')
    s_params = sum(p.numel() for p in s_model.parameters() if p.requires_grad)
    print(f'Shape Encoder parameter count: {s_params}')
    d_params = sum(p.numel() for p in d_model.parameters() if p.requires_grad)
    print(f'Density Encoder parameter count: {d_params}')
    
    train_loss_hist = []
    start_e = 0
    
    if args.load != '':
        if args.load[-4:] != '.pth': # must be .pth
            args.load += '.pth'
        checkpoint = torch.load(args.direct+'Models/'+args.load, weights_only=False)
        
        try:
            fp_model.load_state_dict(checkpoint['state_dict'])
            train_loss_hist = checkpoint['train_losses']
            start_e = checkpoint['epoch']
            args.epochs += start_e
            
            print(f'Successfully loaded {args.load}!!!')
        except:
            print(f'Something went wrong loading {args.load}, starting new!')
            
    criterion = nn.L1Loss()

    train_it = True
    if train_it:
        m_opt = optim.AdamW(fp_model.parameters(),lr=args.lr,weight_decay=args.decay)
        m_sch = optim.lr_scheduler.ReduceLROnPlateau(m_opt, 'min', patience=5)
        
        s_opt = optim.AdamW(s_model.parameters(),lr=args.lr/10,weight_decay=args.decay)
        s_sch = optim.lr_scheduler.ReduceLROnPlateau(s_opt, 'min', patience=5)
        
        d_opt = optim.AdamW(d_model.parameters(),lr=args.lr/10,weight_decay=args.decay)
        d_sch = optim.lr_scheduler.ReduceLROnPlateau(d_opt, 'min', patience=5)
        
        fp_model, s_model, d_model, losses = train(fp_model, s_model, d_model, train_loader, 
                                                   criterion, m_opt, s_opt, d_opt, args.noise,
                                                   m_sch, s_sch, d_sch, device, args.pint, 
                                                   start_e, args.epochs)
        
        state = {
            'state_dict': fp_model.state_dict(),
            'epoch': args.epochs,
            'train_losses': train_loss_hist + losses,
        }
        
        path = args.direct+'Models/'+args.name+'.pth'
        torch.save(state,path)
        
        state = s_model.state_dict()
        path = args.direct+'Models/'+args.name+'_shape_encoder.pth'
        torch.save(state,path)
        
        state = {
            'state_dict': d_model.state_dict(),
        }
        path = args.direct+'Models/'+args.name+'_density_encoder.pth'
        torch.save(state,path)
        print(f'saved {path}')

            
    trn_loss, trn_preds, trn_targs, trn_w_preds, trn_w_targs = evaluate(
        fp_model, s_model, d_model, train_loader, criterion, device)
    avg_loss, avg_preds, avg_targs, avg_w_preds, avg_w_targs = evaluate(
        fp_model, s_model, d_model, test_loader, criterion, device)
    
    weibull_results = {
        "Train": (trn_w_preds, trn_w_targs, trn_loss),
        "Test": (avg_w_preds, avg_w_targs, avg_loss)
    }
    
    results = {
        "Train": (trn_preds, trn_targs, trn_loss),
        "Test": (avg_preds, avg_targs, avg_loss)
    }
        
        ##%

    
    if args.visual:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with 3 subplots
        x = np.linspace(0, 0.5, 100)
    
        for ax, (title, (preds, targs, loss)) in zip(axes, weibull_results.items()):
            ax.scatter(targs, preds)
            ax.set_ylabel("Prediction")
            ax.set_xlabel("Target Probability")
            ax.set_ylim((0, 0.5))
            ax.set_xlim((0, 0.5))
            ax.plot(x, x, color="gray", linestyle=":", alpha=0.8)
            ax.set_title(f"{title} Set\nLoss: {loss:.4f}")
    
        plt.tight_layout()
        plt.show()
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
        x = np.linspace(0, 250000, 1000)
        for ax, (title, (preds, targs, loss)) in zip(axes2, results.items()):
        
            ax.scatter(targs, preds, s=100)
            ax.set_ylabel("Predicted Fatigue Life")
            ax.set_xlabel("Actual Fatigue Life")
            ax.set_title(f"{title} Set\nLoss: {loss:.4f}")
            ax.plot(x, x, color="gray", linewidth=2, linestyle=":", alpha=0.8)
            ax.set_xlim((0,250000))
            ax.set_ylim((0,250000))
            
        plt.tight_layout()
        plt.show()   
    
    pred_fatigue = {}
    
    if args.save_preds:
        
        ###
        # Removed data: R15 08, 05, 06 R, 16 R4, 04 R, 11 R4, 07 R, 10 R3-4, 
        # MTSFX 11 L, 08 R4, 19 L2 L4 R3, 
        ####
        # I really would like to only save this into the _raw.npy if the compressed exists
        # need to change inp_sleuth.py
        
        directory = args.direct + 'Runner/'
        dataset = FailDataset(device, directory, load_cases, seq_len,1,16)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        fp_model.eval()
        s_model.eval()
        d_model.eval()
        with torch.no_grad():
            for X_tensor, element_data, point_cloud, y_tensor, fatigue_life in test_loader:
                
                # encode shape and density
                
                shape_enc = s_model.encode(point_cloud).unsqueeze(1)
                
                encoding_tensor = d_model.encode(element_data).permute(0,2,1)
                
                output, pred = fp_model(X_tensor, encoding_tensor, shape_enc)
                
                plt.plot(output.squeeze(0).cpu().numpy())
                
                pred_value = pred.cpu().numpy()[0]
                
                # Store in dictionary
                pred_fatigue[y_tensor] = pred_value
            plt.ylabel('failure probability')
            plt.xlabel('log(cycles)')
            plt.title(f'Runner prediction data')
            plt.show()  
        
