# -*- coding: utf-8 -*-

"""
Created on Thu Jan 18 15:51:47 2024

@author: Andrew R Wilzman
"""

import numpy as np
import argparse
import os
import h5py
import torch
import arw_training_turing as trn
import networks
from tabulate import tabulate

def save_losses_h5(data_dir, model_name, loop, losses):
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    loss_data_path = f'{data_dir}ae_{model_name}_{loop}_losses.h5'
    with h5py.File(loss_data_path, 'w') as f:
        f.create_dataset('losses', data=losses)
    print(f'Loss data saved as: {loss_data_path}')
    
def extract_state_lists(state_dict, layer_prefixes):
    state_lists = []

    for layer_prefix in layer_prefixes:
        layer_list = [key for key in state_dict.keys() if layer_prefix in key and 'weight' in key]
        state = []

        for idx, key in enumerate(layer_list):
            weight_size = state_dict[key].size()
            state.append([weight_size[1], weight_size[0]])

        state_lists.append(state)

    return state_lists

def grow_network(network, losses, thresh_scale=0.8, width_scale=0.2):
    losses = torch.tensor(losses).detach().cpu()
    quarter_size = len(losses) // 4
    first_quarter_losses = losses[quarter_size:2 * quarter_size]
    third_quarter_losses = losses[2 * quarter_size:3 * quarter_size]
    fourth_quarter_losses = losses[3 * quarter_size:]

    # Calculate mean losses for each quarter
    first_quarter_max = first_quarter_losses.max()
    third_quarter_mean = third_quarter_losses.mean()
    fourth_quarter_mean = fourth_quarter_losses.mean()

    cond = ((fourth_quarter_mean > third_quarter_mean * thresh_scale) and
            (fourth_quarter_mean < third_quarter_mean / thresh_scale) and
            (fourth_quarter_mean < first_quarter_max * thresh_scale))
    if cond:
        print('Growing the network!')
        for layer_type in range(1, 5):
            feat_list = []
            
            for mo in network.get_layer_list(layer_type):
                feat_list.extend([mo.in_features, mo.out_features])
                
            a = feat_list[0]
            b = feat_list[-1] if len(feat_list) < 4 else feat_list[3]
            
            new_width = int(a * width_scale + b * (1 - width_scale))
            
            network.add_layer(layer_type, 0)
            network.change_width(layer_type, 0, new_width)
        
    return network

#Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str,default='')
    
    parser.add_argument('-a','--autoencode', action='store_true')
    parser.add_argument('--vae', action='store_true')
    parser.add_argument('-d','--diffuse', action='store_true')
    parser.add_argument('-g','--gan', action='store_true')
    
    parser.add_argument('--seed', type=int,default=0)
    parser.add_argument('-e','--epochs', type=int,default=0)
    parser.add_argument('-t','--traintime', type=int,default=10)
    parser.add_argument('-i','--init', type=int,default=1)
    parser.add_argument('--grow', action='store_true')
    parser.add_argument('--chpt',type=int,default=0)
    parser.add_argument('--grow_thresh',type=float,default=0.8)
    parser.add_argument('--grow_width',type=float,default=0.2)
    parser.add_argument('-lr', type=float,default=1e-3)
    parser.add_argument('--decay', type=float,default=1e-6)
    parser.add_argument('--chkdecay',type=float,default=0.95)
    parser.add_argument('--batch', type=int,default=1)
    parser.add_argument('--eval_bs', type=int, default=8, help='eval batch size')
    parser.add_argument('--pint', type=int,default=0)
    parser.add_argument('--noise', type=int,default=3)
    parser.add_argument('--hidden1', type=int,default=512)
    parser.add_argument('--hidden2', type=int,default=128)
    parser.add_argument('--hidden3', type=int,default=128)

    parser.add_argument('--name', type=str,default='')
    parser.add_argument('--loadgen', type=str,default='')
    parser.add_argument('--loaddis', type=str,default='')
    parser.add_argument('--pc_gen', type=int,default=0)
    
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--numpoints', type=int,default=2048)
    parser.add_argument('-v','--visual', action='store_true')
    parser.add_argument('-n','--network', type=str, choices=['trs', 'fold'],
                        help='Network call sign')
    
    args = parser.parse_args(['--direct','A:/Work/','-n','fold',
                              '-v',
                              '-g',
                              #'--grow',
                              #'--grow_thresh','0.9',
                              '-i','1',# 3 different layer start combos
                              '--batch','32',
                              '-lr','1e-2','--decay','1e-6',
                              '-e','0',
                              '-t','200',
                              '--pint','1',
                              '--chpt','0',
                              '--cycles','1',
                              '--noise','4',
                              '--name','_',
                              '--pc_gen','20',
                              '--loadgen','diff_med_fold_512_128_128',
                              '--loaddis',''])
                    
    #Initialize vars
    if args.loadgen != '':
        if args.loadgen[-4:] != '.pth': # must be .pth
            args.loadgen += '.pth'
    if args.seed == 0:
        args.seed = torch.randint(10, 8545, (1,)).item() 
    num_points = args.numpoints
    directory = args.direct+'Data/'
    epochs = args.epochs
    learning_rate = args.lr
    wtdecay = args.decay
    batch_size = args.batch
    print_interval = args.pint
    cycles = args.cycles
    
    if epochs == 0:
        epochs = -args.traintime
        if args.chpt > 0:
            loops = args.chpt + 1
        else:
            loops = 1
    if loops > 1:
        epochs = epochs // loops
        
    if print_interval == 0:
        print_interval = max(1,abs(epochs) // 10)
    
    samps = os.listdir(directory+'Compressed/')
    data = []
    
    # Take inventory and read in sample data
    # Remove metatarsals 1 and 5
    xout = ['L1', 'L5', 'R1', 'R5']
        
    for s in samps:
        s_folder = f'{directory}Compressed/{s}'
        MTs = os.listdir(s_folder)
        for MT in MTs:
            if not any(x in MT for x in xout):
                h5_file = f'{directory}Compressed/{s}/{MT}'
                data.append(h5_file)
                    
    with h5py.File(h5_file, 'r') as hf:
        bone = hf['Surface'][:]
        #dense_bone = hf['Pointcloud'][:]
        MTno = hf['MTno']
        side = hf['Side']
    
    # Network setup
    if torch.cuda.is_available():
        print('CUDA available')
        print(torch.cuda.get_device_name(0))
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        print('CUDA *not* available')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    
    # Initialize state
    if args.network == 'trs':
        input_dim = 3
        input2_dim = args.hidden3+3
    elif args.network == 'fold':
        input_dim = 12
        input2_dim = args.hidden1
        
    if args.loadgen != '':
        # Extract previous layer states
        state_dict = torch.load(f'{args.direct}Models/{args.loadgen}')
        state = extract_state_lists(state_dict, ['e_layers1','e_layers2','d_layers1','d_layers2'])
        
    if args.network == 'trs':
        network = networks.arw_TRSNet(args.hidden1,args.hidden3).to(device)
    elif args.network == 'fold':
        network = networks.arw_FoldingNet(args.hidden1,args.hidden3).to(device)
        
    if n_gpus > 1:
        network = torch.nn.DataParallel(network)
        
    if args.loadgen != '':
        network.load_state_dict(torch.load(f'{args.direct}Models/{args.loadgen}'))
        
    if args.visual: #see example input 
        import open3d as o3d
        import pointcloud_handler as pch
        import pandas as pd
        def set_point_cloud_color(point_cloud, color):
            point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(point_cloud.points), 1)))
        color1 = np.array([0.229, 0.298, 0.512])
        color2 = np.array([0.728, 0.440, 0.145])
        color3 = np.array([0.598, 0.770, 0.344]) 
        bonetest = torch.FloatTensor(bone).to(device)

        # Prepare the original, reconstructed, and fake point clouds
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud1.points = o3d.utility.Vector3dVector(bonetest.cpu().detach().numpy())
        set_point_cloud_color(point_cloud1, color=color1)
    
        # Generate fake point clouds using the network
        noise1 = torch.randn(1, args.hidden1, device=device)
        with torch.no_grad():
            fake1 = network.decode(noise1, num_points)
            test = network.encode(bonetest.unsqueeze(0))
            fake = network.decode(test, num_points)
    
        # Reconstructed point cloud
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud2.points = o3d.utility.Vector3dVector(fake.cpu().detach().numpy().squeeze(0))
        set_point_cloud_color(point_cloud2, color=color2)
    
        # Fake point cloud
        point_cloud3 = o3d.geometry.PointCloud()
        point_cloud3.points = o3d.utility.Vector3dVector(fake1.cpu().detach().numpy().squeeze(0))
        set_point_cloud_color(point_cloud3, color=color3)
    
        # Display all point clouds in a single window
        o3d.visualization.draw_geometries([point_cloud2, point_cloud3])
        
        
        if args.pc_gen > 0:
            good_set = []
            vnov_set = []
            bad_set = []
            noise = torch.randn(args.pc_gen, test.shape[1], device=device)
            with torch.no_grad():
                fake = network.decode(noise, num_points)
            fake = fake.cpu().detach().numpy()
     
            # Save point clouds to CSV
            for i in range(args.pc_gen):
                # Flatten each generated point cloud (num_points, 3)
                points = fake[i]
                points = pd.DataFrame(points, columns=['x', 'y', 'z'])
                #points, _ = pch.inc_PCA(points)
                vnv = pch.create_stl(points,
                                     f'{args.direct}Data/Generated/unprocessed/{args.loadgen[:-4]}/{args.loadgen[:-4]}_{i}.stl',
                                     16,False)
                if vnv > 0:
                    good_set.append(noise)
                elif vnv == 0:
                    vnov_set.append(noise)
                else:
                    bad_set.append(noise)                   
            bad_set = np.array([n.detach().cpu().numpy() for n in bad_set])
            vnov_set = np.array([n.detach().cpu().numpy() for n in vnov_set])
            good_set = np.array([n.detach().cpu().numpy() for n in good_set])
            
    if args.grow:
        perc_thresh = int(args.grow_thresh*100)
        perc_width = int(args.grow_width*100)
        model_name = f'{args.name}_{args.hidden1}_{perc_thresh}_{perc_width}'
    else:
        model_name = f'{args.name}_{args.hidden1}_{args.hidden2}_{args.hidden3}'
    
#%%
    if args.diffuse or args.autoencode or args.vae:
        loss_function = trn.Chamfer_Loss()
                      
        if args.autoencode:
            if args.diffuse:
                print_interval = max(1,print_interval // 2)
                
            for loop in range(loops):
                network,losses = trn.train_autoencoder(
                    data,network,epochs,learning_rate,wtdecay,batch_size,
                    loss_function,print_interval,device,num_points,cycles)
                
                loss_file = f'{args.direct}Metrics/'
                save_losses_h5(loss_file, model_name, loop, losses)
                
                # save networks                
                torch.save(network.state_dict(),f'{args.direct}Models/ae_{model_name}_{loop}.pth')
                print(f'Model saved as: {args.direct}Models/ae_{model_name}_{loop}.pth')
                
                chf_loss,jsd = trn.model_eval_chamfer(data, network.to(device), 
                                                      num_points, device, batch_size=args.eval_bs)
                
                # Create and save metric table 
                table = [["Metric", "Value"],
                         ["Chamfer Loss", f'{chf_loss:.3f}'],
                         ["JSD Loss", f'{jsd:.3f}'],
                         ["Epochs", f'{epochs}'],
                         ["Learn Rate", f'{learning_rate}'],
                         ["Decay", f'{args.decay}'],
                         ["Batch Size", f'{args.batch}']]
                
                
                print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
                
                cycles += 1
                
                if args.grow:
                    network = grow_network(network,losses,args.grow_thresh,args.grow_width)
                    network = network.to(device)
                    
            learning_rate = learning_rate * args.chkdecay
            filename = f'{args.direct}Metrics/ae_{model_name}.txt'
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
                
        if args.vae:
            cycles = args.cycles
            for loop in range(loops):
                network,losses = trn.train_vae(data,network,epochs, 
                                               learning_rate,wtdecay,batch_size, 
                                               loss_function,print_interval, 
                                               device,num_points,cycles)
                torch.save(network.state_dict(),f'{args.direct}Models/Vae_{model_name}_{loop}.pth')
                print(f'Model saved as: {args.direct}Models/Vae_{model_name}_{loop}.pth')
                
                chf_loss,jsd = trn.model_eval_chamfer(data, network.to(device),
                                                      num_points, device, batch_size=args.eval_bs)
                
                # Create and save metric table 
                table = [["Metric", "Value"],
                         ["Chamfer Loss", f'{chf_loss:.3f}'],
                         ["JSD Loss", f'{jsd:.3f}'],
                         ["Epochs", f'{epochs}'],
                         ["Learn Rate", f'{learning_rate}'],
                         ["Decay", f'{args.decay}'],
                         ["Batch Size", f'{args.batch}']]
                
                print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
                
                cycles += 1
                
                if args.grow:
                    network = grow_network(network,losses,args.grow_thresh,args.grow_width)
                    network = network.to(device)
                
            learning_rate = learning_rate * args.chkdecay 
            filename = f'{args.direct}Metrics/Vae_{model_name}.txt'
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
                
        if args.diffuse:
            cycles = args.cycles
            seg_epochs = epochs//args.noise # segment epoch length for
            for n in range(args.noise+1):             # gradually increasing noise
                print(f'training {n+1} / {args.noise+1}')
                network,losses = trn.train_diffusion(data,network,seg_epochs,learning_rate,wtdecay,
                                                     batch_size,loss_function,print_interval,
                                                     device,n+1,num_points,cycles)
                learning_rate = learning_rate * args.chkdecay
                
            # save network
            torch.save(network.state_dict(),f'{args.direct}Models/diff_{model_name}.pth')
            print(f'Model saved as: {args.direct}Models/diff_{model_name}.pth')
            
            chf_loss,jsd = trn.model_eval_chamfer(data, network, num_points, device, args.eval_bs)
            
            # Create and save metric table 
            table = [["Metric", "Value"],
                     ["Chamfer Loss", f'{chf_loss:.3f}'],
                     ["JSD Loss", f'{jsd:.3f}'],
                     ["Epochs", f'{epochs}'],
                     ["Learn Rate", f'{learning_rate}'],
                     ["Decay", f'{args.decay}'],
                     ["Batch Size", f'{args.batch}']]
            
            learning_rate = learning_rate * args.chkdecay
            
            print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
        
            filename = f'{args.direct}Metrics/diff_{model_name}.txt'
            
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
            
    # GAN training
    if args.gan:
        
        loss_function = trn.GAN_Loss()
        Dnet = networks.jarvis(bone.shape[0]).to(device)
        
        if args.loaddis != '':
            if args.loaddis[-4:] != '.pth':
                args.loaddis += '.pth'
            Dnet.load_state_dict(torch.load(f'{args.direct}Models/{args.loaddis}'))
            
        gnet,gloss,dnet,dloss = trn.train_GD(data,network,Dnet,args.hidden1,epochs,
                                             learning_rate,wtdecay,batch_size,
                                             loss_function,print_interval,device)
        
        torch.save(gnet.state_dict(),f'{args.direct}Models/gangen_{model_name}.pth')
        print(f'Generator saved as: {args.direct}Models/gangen_{model_name}.pth')
        
        torch.save(dnet.state_dict(),f'{args.direct}Models/Discrim_{model_name}.pth')
        print(f'Discriminator saved as: {args.direct}Models/Discrim_{model_name}.pth')
        
        chf_loss,jsd = trn.model_eval_chamfer(data, gnet, num_points, device, args.eval_bs)
        
        # Create and save metric table 
        table = [["Metric", "Value"],
                 ["Chamfer Loss", f'{chf_loss:.3f}'],
                 ["JSD Loss", f'{jsd:.3f}'],
                 ["Epochs", f'{epochs}'],
                 ["Learn Rate", f'{args.lr}'],
                 ["Decay", f'{args.decay}'],
                 ["Batch Size", f'{args.batch}']]
        
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
        
        filename = f'{args.direct}Metrics/gangen_{model_name}.txt'
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))
            