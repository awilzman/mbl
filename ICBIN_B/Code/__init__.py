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
from sklearn.neighbors import NearestNeighbors


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

def compute_knn_graph(positions, k=16):
    neighbors = NearestNeighbors(n_neighbors=k+1).fit(positions)
    distances, indices = neighbors.kneighbors(positions)
    edge_index = []
    
    for i in range(positions.shape[0]):
        for j in range(1, k+1):  # Start from 1 to skip the point itself
            edge_index.append([i, indices[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

#Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', type=str,default='')
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
    parser.add_argument('--hidden3', type=int,default=1024)
    parser.add_argument('--name', type=str,default='')
    parser.add_argument('--loadgen', type=str,default='')
    parser.add_argument('--loadclass', type=str,default='')
    parser.add_argument('--loaddis', type=str,default='')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--numpoints', type=int,default=2048)
    parser.add_argument('-d','--diffuse', action='store_true')
    parser.add_argument('-g','--gan', action='store_true')
    parser.add_argument('-a','--autoencode', action='store_true')
    parser.add_argument('--vae', action='store_true')
    parser.add_argument('-v','--visual', action='store_true')
    parser.add_argument('-p','--pretrain', action='store_true')
    parser.add_argument('-n','--network', type=str, choices=['trs', 'fold', 'mlp'],
                        help='Network call sign')
    
    args = parser.parse_args(['--direct','../','-n','fold',
                              '-a',
                              #'--grow',
                              '--grow_thresh','0.9',
                              '-i','2',
                              '-a',
                              '-lr','1e-4','--decay','1e-5',
                              '-e','0',
                              '-t','600',
                              '--pint','1',
                              '--chpt','0',
                              '--cycles','1',
                              '--noise','5',
                              '--name','',
                              '--loadgen','',
                              '--loadclass','',
                              '--loaddis',''])
                    
    #Initialize vars
    if args.loadgen != '':
        if args.loadgen[-4:] != '.pth': # must be .pth
            args.loadgen += '.pth'
    if args.loadclass != '':
        if args.loadclass[-4:] != '.pth': # must be .pth
            args.loadclass += '.pth'
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
    elif args.network == 'mlp':
        input_dim = 3
        input2_dim = args.hidden3
        
    if args.loadgen != '':
        # Extract previous layer states
        state_dict = torch.load(f'{args.direct}Models/{args.loadgen}')
        state = extract_state_lists(state_dict, ['e_layers1','e_layers2','d_layers1','d_layers2'])
    elif not args.grow:
        if args.init == 1:
            state = [[(input_dim,args.hidden3)],
                     [(input2_dim,args.hidden1)],
                     [(args.hidden1+2,3)],
                     [(args.hidden1+3,3)]]
        elif args.init == 2:
            state = [[(input_dim,64),(64,args.hidden3)],
                     [(input2_dim,args.hidden1*2),(args.hidden1*2,args.hidden1)],
                     [(args.hidden1+2,args.hidden3),(args.hidden3,3)],
                     [(args.hidden1+3,args.hidden3),(args.hidden3,3)]]
        elif args.init == 3:
            state = [[(input_dim,64),(64,128),(128,256),(256,args.hidden3)],
                     [(input2_dim,args.hidden1*4),(args.hidden1*4,args.hidden1*2),(args.hidden1*2,args.hidden1),(args.hidden1,args.hidden1)],
                     [(args.hidden1+2,args.hidden3),(args.hidden3,args.hidden3//2),(args.hidden3//2,args.hidden3//16),(args.hidden3//16,3)],
                     [(args.hidden1+3,args.hidden3),(args.hidden3,args.hidden3//2),(args.hidden3//2,args.hidden3//16),(args.hidden3//16,3)]]
    else:
        state = None
        
    if args.network == 'trs':
        network = networks.arw_TRSNet(args.hidden1,args.hidden3,state).to(device)
    elif args.network == 'fold':
        network = networks.arw_FoldingNet(args.hidden1,args.hidden3,state).to(device)
    elif args.network == 'mlp':
        network = networks.arw_MLPNet(args.hidden1,args.hidden3,state).to(device)
            
    if n_gpus > 1:
        network = torch.nn.DataParallel(network)
        
    if args.loadgen != '':
        network.load_state_dict(torch.load(f'{args.direct}Models/{args.loadgen}'))
        
    if args.visual: #see example input 
        import open3d as o3d
        def set_point_cloud_color(point_cloud, gray_value):
            # Ensure the color array matches the number of points
            gray_color = np.tile([gray_value, gray_value, gray_value], (len(point_cloud.points), 1))
            point_cloud.colors = o3d.utility.Vector3dVector(gray_color)
        
        point_cloud = o3d.geometry.PointCloud()
        bonetest = torch.FloatTensor(bone).to(device)
        bone_knn = compute_knn_graph(bone).to(device)
        
        with torch.no_grad():
            test = network.encode(bonetest.unsqueeze(0), bone_knn)
            fake = network.decode(test, num_points)
        fake = fake.cpu().detach().numpy()
        
        # Original bone point cloud
        point_cloud.points = o3d.utility.Vector3dVector(bonetest.cpu().detach().numpy())
        set_point_cloud_color(point_cloud, gray_value=0.5)
        o3d.visualization.draw_geometries([point_cloud])
        
        # Fake point cloud
        point_cloud.points = o3d.utility.Vector3dVector(fake.squeeze(0))
        set_point_cloud_color(point_cloud, gray_value=0.4)
        o3d.visualization.draw_geometries([point_cloud])
        
        for i in range(2):
            noise = torch.randn(test.shape[0], 1, test.shape[2], device=device)
            with torch.no_grad():
                fake = network.decode(noise, num_points)
            fake = fake.cpu().detach().numpy()
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(fake.squeeze(0))
            set_point_cloud_color(point_cloud, gray_value=0.3)
            o3d.visualization.draw_geometries([point_cloud])
        
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
                
                # save networks                
                torch.save(network.state_dict(),f'{args.direct}Models/ae_{model_name}_{loop}.pth')
                print(f'Model saved as: {args.direct}Models/ae_{model_name}_{loop}.pth')
                
                chf_loss,jsd = trn.model_eval_chamfer(data, network.to(device), 
                                                      num_points, device, batch_size=args.eval_bs)
                
                # Create and save metric table 
                table = [["Metric", "Value"],
                         ["Chamfer Loss", f'{chf_loss:.3f}'],
                         ["JSD Loss", f'{jsd:.3f}'],
                         #["Classifier Loss", f"{class_loss_}"],
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
            
        gnet,gloss,dnet,dloss = trn.train_GD(data,network,Dnet,args.hidden3,epochs,
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
