import torch
import argparse
import os
import inp_sleuth as inpsl
import density_networks as dnets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--direct', type=str, default='')
    parser.add_argument('-l','--load', type=str, default='')
    
    args = parser.parse_args(['-d', 'A:/Work/',
                              '-l','det_den3'
                              ])
    
    if torch.cuda.is_available():
        print('CUDA available')
        print(torch.cuda.get_device_name(0))
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")
    else:
        print('CUDA *not* available')
        
    if args.load != '':
        if args.load[-4:] != '.pth': # must be .pth
            args.load += '.pth'
            
    fab_data = args.direct+'Data/inps/Fabricated/'
    inp_files = [f for f in os.listdir(fab_data) if f.endswith('.inp')]
    
    all_data = {}
    for inp_file in inp_files:
        inp_path = os.path.join(fab_data, inp_file)
        parser = inpsl.AbaqusInpParser(inp_path)
        element_data = parser.process_inp_file()
        all_data[inp_file].append(element_data)