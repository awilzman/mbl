import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import inp_sleuth as inpsl
import density_networks as dnets
import density_training as dtrn
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

def repack_nodes(nodes):
    nset_upd = []
    temp = [] 
    nodes = sorted(nodes, key=int)
    nodes = [str(item) for item in nodes]
    for i, node_id in enumerate(nodes):
        temp.append(node_id.strip())
        if i % 16 == 15:
            nset_upd.append(', '.join(temp))
            temp = []
            
    if temp:
        nset_upd.append(', '.join(temp))
    return nset_upd

def write_inp(output,output_file):
    # Writes a list (output) to a file (output_file)
    if '.' not in output_file:
        output_file += '.inp' # Default to .inp file if not specified
    output = [str(elem) for elem in output]
    with open(output_file, 'w') as out_file:
        for line in output:
            out_file.write(line+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--direct', type=str, default='')
    parser.add_argument('-l','--load', type=str, default='')
    parser.add_argument('--hidden1', type=int, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('-b','--bidir', action='store_true')
    parser.add_argument('--savevtk', action='store_true')
    parser.add_argument('-v','--visualize', action='store_true')
    parser.add_argument('-n','--noise', type=float, default=1e-3)
    parser.add_argument('--matnum', type=int, default=100)
    parser.add_argument('--axial_loads', default=[100,200,300])
    parser.add_argument('--bend_loads', default=[100,200,300])
    parser.add_argument('--bend_angle', type=int, default=30)
    
    args = parser.parse_args(['-d', 'A:/Work/',#'-v',
                              #'-b',
                              '-l','crimp',
                              '--hidden1', '32',
                              '--layers', '6'
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
    fab_data = os.path.join(args.direct, 'Data/inps/Fabricated/geo_only')
    inp_files = [f for f in os.listdir(fab_data) if f.endswith('.inp')]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = dnets.tet10_encoder(args.hidden1, args.layers, args.bidir).to(device)
    densifier = dnets.tet10_densify(args.hidden1).to(device)
    
    checkpoint = torch.load(os.path.join(args.direct, 'Models', args.load))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    densifier.load_state_dict(checkpoint['densifier_state_dict'])
    scale_factor= checkpoint['scale_factor']
    
    #densify all inp_files
    for inp_file_idx, inp_file in enumerate(inp_files):
        #grab element data
        inp_path = os.path.join(fab_data, inp_file)
        
        inp_part = ['*Part, name=PART-1']
        inp_nodes = ['*NODE']
        inp_elements = inpsl.read_inp(inp_path,'*ELEMENT','*',True)
        inp_surf_elements = inpsl.read_inp(inp_path,'*SURFACE','*',True)
        inp_orient = ['*Orientation, name = Ori-1','1., 0., 0., 0., 1., 0.','1, 0.']
        inp_matsets = []
        for i in range(args.matnum):
            inp_matsets.append(f'*ELSET, ELSET=MAT{i}')
            inp_matsets.append(f'*SOLID SECTION, ELSET=MAT{i},orientation=Ori-1, MATERIAL=MAT{i}')
        inp_act_ele = ['*ELSET, ELSET=Active_Elements']
        inp_endpart = ['*End Part','*Assembly, name=Assembly','*Instance, name=PART-1-1, part=PART-1',
                       '*End Instance']
        inp_surf_nodes = inpsl.read_inp(inp_path,'*NSET, NSET=NS_Surface','*',True)
        inp_load_node = ['*NSET, NSET=Load_Node, instance=PART-1-1']
        inp_bend_node = ['*NSET, NSET=Bend_Load_Node, instance=PART-1-1']
        inp_fixed_nodes = ['*NSET, NSET=Fixed_Nodes, instance=PART-1-1']
        inp_active_nodes = ['*NSET, NSET=Active_Nodes, instance=Part-1-1']
        inp_coupling = ['*Coupling, constraint name=Constraint-1, ref node=Load_Node, surface=PART-1-1.SURFACE, influence radius=0.005',
                        '*Kinematic']
        inp_endass = ['*End Assembly']
        inp_mats = []
        for i in range(args.matnum):
            inp_mats.append(f'*MATERIAL, NAME=MAT{i}')
            inp_mats.append('*DENSITY')
            inp_mats.append('*ELASTIC, type=ENGINEERING CONSTANTS')
        inp_bc = ['*Boundary','Fixed_Nodes, 1, 1','Fixed_Nodes, 2, 2','Fixed_Nodes, 3, 3']
        inp_first_step = ['*Step, NAME = ','*STATIC','*Cload',
                         '*Restart, write, frequency=0','*Output, field','*Node Output, nset=Active_Nodes',
                         'CF, RF, U','*Element Output, elset=PART-1-1.Active_Elements, directions=YES',
                         'E, EVOL, MISES, MISESMAX, S','*Output, history','*Contact Output',
                         'CSMAXSCRT,','*End Step']
        inp_next_step = ['*Step, NAME = ','*STATIC','*Cload','*End Step']
        inp_newloadnode_step = ['*Step, NAME = ','*STATIC','*Cload, OP=NEW','*End Step']
        
        
        inp_parser = inpsl.AbaqusInpParser(inp_path)
        element_data = inp_parser.process_inp_file()
        node_data = inp_parser.nodes
        nodes = np.array(list(node_data.values()))*1e-3# mm -> m
        node_ids = np.array(list(node_data.keys())).reshape(-1, 1)
        node_data = np.hstack((node_ids, nodes))
        for node in node_data:
            inp_nodes.append(f'\t{int(node[0])}, {node[1]:.9e}, {node[2]:.9e}, {node[3]:.9e}')
        element_data[:,:-2] = element_data[:,:-2]*1e-3# mm -> m 
        
        #guess densities, inp_parser has 0s in place of density if not read
        X = torch.FloatTensor(element_data[:,:-1]) 
        sorted_indices = torch.argsort(X[:, 2])
        sorted_indices = sorted_indices[torch.argsort(X[sorted_indices, 1])]
        sorted_indices = sorted_indices[torch.argsort(X[sorted_indices, 0])]
        X = X[sorted_indices].unsqueeze(0).to(device)
        
        with torch.no_grad():
            encoded, l = encoder(X)
            encoded = encoded + torch.randn_like(encoded) * args.noise
            E_out = densifier(X, encoded)
            
        X = X.detach().cpu().numpy().squeeze(0)
        E_out = E_out.detach().cpu().numpy().squeeze(0)
        
        if args.visualize:
            dtrn.show_bone([X,E_out],scale_factor)
            
        #calculate real e11 with scale factor and assign anisotropy
        #create args.matnum bins of material definitions
        #assign each element ID to a bin
        #write material assignments expicitly
        
        E_out = E_out * scale_factor
        min_E = max(2.39e+06**1.15,min(E_out))
        min_d = max(15,min(E_out)**(1/1.15)/2.39e6)
        
        bins = np.logspace(np.log10(min_E), np.log10(max(E_out)), num=args.matnum)
        
        mat_vals = np.zeros((bins.shape[0], 9))
        densities = np.linspace(min_d,max(E_out)**(1/1.15)/2.39e6, num=args.matnum)
        
        ###
        # if z is max range: E11, E22, E33, v12, v23, v31, G13, G31, G23
        # if y ... 
        # if x ...
        ###
        coefficients = {
            'z': [0.574, 0.577, 1.0, 0.427, 0.234, 0.405, 0.195, 0.216, 0.265],
            'y': [0.574, 1.0, 0.577, 0.405, 0.427, 0.234, 0.216, 0.195, 0.265],
            'x': [1.0, 0.577, 0.574, 0.405, 0.234, 0.427, 0.265, 0.216, 0.195],
        }
        
        rangx = max(nodes[:,0])-min(nodes[:,0])
        rangy = max(nodes[:,1])-min(nodes[:,1])
        rangz = max(nodes[:,2])-min(nodes[:,2])
        
        largest = max(rangx, rangy, rangz)
        bins = bins.squeeze(-1)
        if rangz == largest:
            layout = [2,1,0]
            mat_vals[:, 0] = coefficients['z'][0] * bins
            mat_vals[:, 1] = coefficients['z'][1] * bins
            mat_vals[:, 2] = coefficients['z'][2] * bins
            mat_vals[:, 3] = coefficients['z'][3]
            mat_vals[:, 4] = coefficients['z'][4]
            mat_vals[:, 5] = coefficients['z'][5]
            mat_vals[:, 6] = coefficients['z'][6] * bins
            mat_vals[:, 7] = coefficients['z'][7] * bins
            mat_vals[:, 8] = coefficients['z'][8] * bins
        
        elif rangy == largest:
            layout = [1,2,0]
            mat_vals[:, 0] = coefficients['y'][0] * bins
            mat_vals[:, 1] = coefficients['y'][1] * bins
            mat_vals[:, 2] = coefficients['y'][2] * bins
            mat_vals[:, 3] = coefficients['y'][3]
            mat_vals[:, 4] = coefficients['y'][4]
            mat_vals[:, 5] = coefficients['y'][5]
            mat_vals[:, 6] = coefficients['y'][6] * bins
            mat_vals[:, 7] = coefficients['y'][7] * bins
            mat_vals[:, 8] = coefficients['y'][8] * bins
        
        elif rangx == largest:
            layout = [0,1,2]
            mat_vals[:, 0] = coefficients['x'][0] * bins
            mat_vals[:, 1] = coefficients['x'][1] * bins
            mat_vals[:, 2] = coefficients['x'][2] * bins
            mat_vals[:, 3] = coefficients['x'][3]
            mat_vals[:, 4] = coefficients['x'][4]
            mat_vals[:, 5] = coefficients['x'][5]
            mat_vals[:, 6] = coefficients['x'][6] * bins
            mat_vals[:, 7] = coefficients['x'][7] * bins
            mat_vals[:, 8] = coefficients['x'][8] * bins
        
        
        axis = layout[0]+1
        bend_dir = layout[1]+1
        
        max_z = max(node_data[:,axis])
        min_z = min(node_data[:,axis])
        
        # If the mean {bend_dir} value in the mid diaphysis 
        # is less than zero,
        # then the bend direction is negative
        
        
        # If there are more nodes on the positive {axis} than negative,
        # then the axial direction is positive, making the load node the most
        # negative point.
        z_pos = [
            node[0] for node in node_data if float(
                node[axis] > max_z - 0.005)]
        z_neg = [
            node[0] for node in node_data if float(
                node[axis] > min_z + 0.005)]
        
        if len(z_pos) > len(z_neg):
            axial_dir = 1
            axial_load_node = int([nd[0] for nd in node_data if nd[axis]==min_z][0])
            
            fixed_nodes = [int(node[0]) for node in node_data if float(node[axis]) > max_z-0.025]
            
            active_nodes = [
                int(node[0]) for node in node_data if float(
                    node[axis]) < max_z - 0.03 and (
                        float(node[axis]) > min_z + 0.015)]
        else:
            axial_dir = -1
            axial_load_node = int([nd[0] for nd in node_data if nd[axis]==max_z][0])
            fixed_nodes = [int(node[0]) for node in node_data if float(node[axis]) < min_z+0.025]
            
            active_nodes = [
                int(node[0]) for node in node_data if float(
                    node[axis]) < max_z - 0.015 and (
                        float(node[axis]) > min_z + 0.03)]
                        
        active_elements = [int(ele.split(',')[0]) for ele in inp_elements[1:] if all(
            node in active_nodes for node in [int(i) for i in ele.split(',')[1:]])]
        
        # If more surface points exist on positive axis bend_dir is positive
        y_check = [float(nd[bend_dir]) for nd in node_data[active_nodes]]
        y_check = sum(y_check)
        
        bend_neg = y_check < 0
        
        # Find bend load node by finding the minimum {bend_dir} point,
        # among the nodes on the distal side,
        # if {bend_dir} is positive, or the maximum if it is negative
        max_y = max(node_data[active_nodes,bend_dir])
        min_y = min(node_data[active_nodes,bend_dir])
        
        if bend_neg:
            bend_load_node = int([nd[0] for nd in node_data if nd[bend_dir]==max_y][0])
        else:
            bend_load_node = int([nd[0] for nd in node_data if nd[bend_dir]==min_y][0])
        
        #start a dictionary, labels 0-199, in which each label will hold a different
        #amount of nodes. Then, loop through each bin and find all elements
        #that are less than the bin value, but have not been assigned a bin yet. 
        #We need to start at the lowest value bin, which is index 0
        ele_set_bins = {i: [] for i in range(len(bins))}
        for i, modulus in enumerate(E_out):
            if modulus < bins[0]:
                bin_idx = 0
            elif modulus >= bins[-1]:  # Check if modulus is greater than or equal to the last bin
                bin_idx = len(bins) - 1
            else:
                bin_idx = int(np.searchsorted(bins, modulus, side='right')[0] - 1)
                
            ele_set_bins[bin_idx].append(i+1)
        c=0
        inp_newmatsets = []
        for stuf in inp_matsets:
            inp_newmatsets.append(stuf)
            if '*ELSET' in stuf:
                node_list = repack_nodes(ele_set_bins[c])
                for stuff in (node_list):
                    inp_newmatsets.append(stuff)
                c+=1
                
        for i in repack_nodes(active_elements):
            inp_act_ele.append(i)
        
        inp_load_node.append(axial_load_node)
        inp_bend_node.append(bend_load_node)
        
        f_fixed_nodes = repack_nodes(fixed_nodes)
        f_active_nodes = repack_nodes(active_nodes)
        
        for i in f_fixed_nodes:
            inp_fixed_nodes.append(i)
        for i in f_active_nodes:
            inp_active_nodes.append(i)
        
        c=0
        inp_materials = []
        for stuf in inp_mats:
            inp_materials.append(stuf)
            if '*DENSITY' in stuf:
                inp_materials.append(f'{densities[c][0]}')
                
            if '*ELASTIC' in stuf:
                string = f'{mat_vals[c,0]}, '
                string += f'{mat_vals[c,1]}, '
                string += f'{mat_vals[c,2]}, '
                string += f'{mat_vals[c,3]}, '
                string += f'{mat_vals[c,4]}, '
                string += f'{mat_vals[c,5]}, '
                string += f'{mat_vals[c,6]}, '
                string += f'{mat_vals[c,7]}, '
                inp_materials.append(string)
                inp_materials.append(f'{mat_vals[c,8]}')
                c+=1
        
        inp_steps = []
        # First axial step
        if axial_dir < 0:
            load = -1 * abs(args.axial_loads[0])
        else:
            load = abs(args.axial_loads[0])
        inp_steps.append(inp_first_step[0] + f'axial_{abs(load)}N')
        inp_steps.append(inp_first_step[1])
        inp_steps.append(inp_first_step[2])
        inp_steps.append(f'Load_Node, {axis}, {load}')
        inp_steps.extend(inp_first_step[3:])
        
        # Remaining axial loads
        for load in args.axial_loads[1:]:
            if axial_dir < 0:
                load = -1 * abs(load)
            else:
                load = abs(load)
            inp_steps.append(inp_next_step[0] + f'axial_{abs(load)}N')
            inp_steps.append(inp_next_step[1])
            inp_steps.append(inp_next_step[2])
            inp_steps.append(f'Load_Node, {axis}, {load}')
            inp_steps.append(inp_next_step[3])
        
        # First bending load
        load1 = abs(np.cos(args.bend_angle) * args.bend_loads[0])
        load2 = abs(np.sin(args.bend_angle) * args.bend_loads[0])
        
        if axial_dir < 0:
            load1 *= -1
        if bend_neg:
            load2 *= -1
        
        inp_steps.append(inp_newloadnode_step[0] + f'deg{args.bend_angle}_{args.bend_loads[0]}N')
        inp_steps.append(inp_newloadnode_step[1])
        inp_steps.append(inp_newloadnode_step[2])
        inp_steps.append(f'Load_Node, {axis}, {load1}')
        inp_steps.append(f'Load_Node, {bend_dir}, {load2}')
        inp_steps.append(inp_newloadnode_step[3])
        
        # Remaining bending loads
        for load in args.bend_loads[1:]:
            load1 = abs(np.cos(args.bend_angle) * load)
            load2 = abs(np.sin(args.bend_angle) * load)
            
            if axial_dir < 0:
                load1 *= -1
            if bend_neg:
                load2 *= -1
            
            inp_steps.append(inp_newloadnode_step[0] + f'deg{args.bend_angle}_{load}N')
            inp_steps.append(inp_newloadnode_step[1])
            inp_steps.append(inp_newloadnode_step[2])
            inp_steps.append(f'Load_Node, {axis}, {load1}')
            inp_steps.append(f'Load_Node, {bend_dir}, {load2}')
            inp_steps.append(inp_newloadnode_step[3])
        
        final_inp = (inp_part+inp_nodes+inp_elements+inp_surf_elements+
                     inp_orient+inp_newmatsets+inp_act_ele+inp_endpart+
                     inp_load_node+inp_bend_node+inp_fixed_nodes+inp_active_nodes+
                     inp_coupling+inp_endass+inp_materials+inp_bc+inp_steps)
        new_inp_file = f'{args.direct}Data/inps/Fabricated/{inp_file_idx}.inp'
        write_inp(final_inp,new_inp_file)