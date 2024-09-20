import torch
import argparse
import os
import inp_sleuth as inpsl
import density_networks as dnets
import numpy as np
import pyvista as pv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--direct', type=str, default='')
    parser.add_argument('-l','--load', type=str, default='')
    parser.add_argument('--hidden1', type=int, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--bidir', action='store_true')
    parser.add_argument('--savevtk', action='store_true')
    parser.add_argument('-v','--visualize', action='store_true')

    args = parser.parse_args(['-d', 'A:/Work/','-v',
                              '-l','arw',
                              '--hidden1', '128',
                              '--layers', '1',
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
    
    all_data = {}
    for inp_file in inp_files:
        #grab element data
        all_data[inp_file] = []
        inp_path = os.path.join(fab_data, inp_file)
        
        inp_parser = inpsl.AbaqusInpParser(inp_path)
        inp_data = inp_parser.read_inp()
        inp_parser.extract_nodes_elements(inp_data)
        element_data = inp_parser.create_element_data()
        
        #guess densities
        X = torch.FloatTensor(element_data[:,:-1]).unsqueeze(0).to(device)
        with torch.no_grad():
            encoded = encoder(X)
            d_out = densifier(X, encoded)
        
        d_out = (d_out * scale_factor).detach().cpu().numpy()
        element_data[:,-1] = d_out.flatten()
            
        all_data[inp_file].append(element_data)
        
        if args.savevtk or args.visualize:
            points = []
            cells = []
            
            for elem in element_data:
                nodes = elem[:30].reshape(10, 3)  # 10 nodes for each tetrahedron
                points.extend(nodes)
                start_idx = len(points) - 10  
                cells.append([10] + list(range(start_idx, start_idx + 10)))
            
            points = np.array(points)
            
            cell_type = np.full(len(cells), pv.CellType.TETRA, dtype=np.int8)
            
            grid = pv.UnstructuredGrid(cells, cell_type, points)
            
            grid.cell_data['Density'] = element_data[:,-1]
            
            if args.savevtk:
                grid.save(f"{fab_data}{inp_file[:-4]}.vtk")
            if args.visualize:
                grid.plot(scalars='Density', show_edges=True, cmap='viridis')
