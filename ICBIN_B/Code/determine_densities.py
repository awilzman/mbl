import torch
import torch.nn as nn
import argparse
import os
import inp_sleuth as inpsl
import density_networks as dnets
import density_training as dtrn
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import seaborn as sns

def plot_weights(model, layer_name):
    """Plot the weights of a specific layer or parameter in the model."""
    # Handle if layer_name is for a specific parameter (e.g., LSTM weights)
    if layer_name in dict(model.named_parameters()):
        layer_weights = dict(model.named_parameters())[layer_name].detach().cpu().numpy()
        
        # Plot the weights
        plt.figure(figsize=(10, 6))
        sns.heatmap(layer_weights, cmap='viridis', annot=False)
        plt.title(f'Weights of {layer_name}')
        plt.xlabel('Neurons')
        plt.ylabel('Inputs')
        plt.show()

    # Handle Sequential layers
    elif layer_name in dict(model.named_modules()):
        layer = dict(model.named_modules())[layer_name]
        
        if isinstance(layer, nn.Sequential):
            # Iterate over sublayers and plot their weights
            for idx, sublayer in enumerate(layer):
                if hasattr(sublayer, 'weight'):
                    layer_weights = sublayer.weight.detach().cpu().numpy()
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(layer_weights, cmap='viridis', annot=False)
                    plt.title(f'Weights of {layer_name}[{idx}] ({sublayer.__class__.__name__})')
                    plt.xlabel('Neurons')
                    plt.ylabel('Inputs')
                    plt.show()
        else:
            raise KeyError(f"Layer '{layer_name}' is not a Sequential or named parameter.")
    else:
        raise KeyError(f"Layer '{layer_name}' not found in model.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--direct', type=str, default='')
    parser.add_argument('-l','--load', type=str, default='')
    parser.add_argument('--hidden1', type=int, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--experts', type=int, default=1)
    parser.add_argument('-b','--bidir', action='store_true')
    parser.add_argument('--savevtk', action='store_true')
    parser.add_argument('-v','--visualize', action='store_true')
    parser.add_argument('-n','--noise', type=float, default=1e-3)

    args = parser.parse_args(['-d', 'A:/Work/','-v',
                              #'-b',
                              '-l','simp2',
                              '--hidden1', '32',
                              '--layers', '4'
                              ])
    
    if torch.cuda.is_available():
        print('CUDA available')
        print(torch.cuda.get_device_name(0))
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        print('CUDA *not* available')
        
    if args.load != '':
        if args.load[-4:] != '.pth':
            args.load += '.pth'
    
    #load fabricated inps
    fab_data = os.path.join(args.direct, 'Data/inps/Fabricated/')
    inp_files = [f for f in os.listdir(fab_data) if f.endswith('.inp')]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = dnets.tet10_encoder(args.hidden1, args.layers, args.bidir).to(device)
    densifier = dnets.tet10_densify(args.hidden1).to(device)
    
    checkpoint = torch.load(os.path.join(args.direct, 'Models', args.load))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    densifier.load_state_dict(checkpoint['densifier_state_dict'])
    
    epochs_trained = checkpoint['epoch']
    train_loss_hist = checkpoint.get('train_losses', [])
    test_loss = checkpoint['testing_loss']
    scale_factor = checkpoint['scale_factor']
    
    plot_weights(encoder, 'lstm.weight_ih_l0')
    plot_weights(encoder, 'lstm.weight_hh_l0')
    plot_weights(encoder, 'fc1')
    all_data = {}
    
    #densify all inp_files
    for inp_file in inp_files:
        #grab element data
        all_data[inp_file] = []
        inp_path = os.path.join(fab_data, inp_file)
        
        inp_parser = inpsl.AbaqusInpParser(inp_path)
        inp_data = inp_parser.read_inp()
        inp_parser.extract_nodes_elements(inp_data)
        element_data = inp_parser.create_element_data()
        
        #guess densities, inp_parser has 0s in place of density if not read
        X = torch.FloatTensor(element_data[:,:-1]*1e-3) # mm -> m 
        sorted_indices = torch.argsort(X[:, 2])
        sorted_indices = sorted_indices[torch.argsort(X[sorted_indices, 1])]
        sorted_indices = sorted_indices[torch.argsort(X[sorted_indices, 0])]
        X = X[sorted_indices].unsqueeze(0).to(device)
        
        with torch.no_grad():
            encoded = encoder(X)
            encoded = encoded + torch.randn_like(encoded) * args.noise
            d_out = densifier(X, encoded)
            
        X = X.detach().cpu().numpy().squeeze(0)
        d_out = d_out.detach().cpu().numpy().squeeze(0)
        
        if args.visualize:
            
            dtrn.show_bone([X,d_out],scale_factor)
