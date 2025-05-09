# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:48:07 2024

@author: arwilzman
"""
import pandas as pd
import numpy as np
import datetime
import os
import argparse
from sklearn.decomposition import IncrementalPCA 

def read_inp(input_file, header='*HEADING', stop_line='bleepbloop', keep=False):
    # Default values result in whole file read, not including heading
    # and only if *HEADING is there in the beginning
    with open(input_file, 'r') as inp_file:
        in_data = False
        data = []
        for line in inp_file:
            line = line.strip()
            if line.startswith(header):
                in_data = True
                if keep:
                    data.append(line)
            elif in_data:
                if line.startswith(stop_line):
                    in_data = False
                else:
                    data.append(line)
    return data

def scale_n_PCA(nodes,scale,layout=[0,1,2]):
    node_id = []
    x = []
    y = []
    z = []
    
    for node in nodes:
        node = node.split(', ')
        node_id.append(int(node[0]))
        x.append(float(node[1]) * scale)
        y.append(float(node[2]) * scale)
        z.append(float(node[3]) * scale)
        
    original_coordinates = np.array([x,y,z])
    
    coordinates = np.array([original_coordinates[layout[0]],
                            original_coordinates[layout[1]],
                            original_coordinates[layout[2]]])
    
    ipca = IncrementalPCA(n_components=3, batch_size=1000)
    ipca.fit(coordinates.T)
    reoriented_coordinates = ipca.transform(coordinates.T)
    new_centroid = np.mean(reoriented_coordinates, axis=0)
    
    reoriented_coordinates[:,0] -= new_centroid[0]
    reoriented_coordinates[:,1] -= new_centroid[1]
    reoriented_coordinates[:,2] -= new_centroid[2]
    
    l = np.array(layout)
    x_s = np.where(l == 0)[0][0]
    y_s = np.where(l == 1)[0][0]
    z_s = np.where(l == 2)[0][0]
    
    x = reoriented_coordinates[:,x_s]
    y = reoriented_coordinates[:,y_s]
    z = reoriented_coordinates[:,z_s]
    
    nodes = np.column_stack((node_id,x,y,z))
    nodes = [f"\t{int(row[0])}, {row[1]:.9e}, {row[2]:.9e}, {row[3]:.9e}" for row in nodes]
    
    return nodes

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
    date = datetime.datetime.now()
    month = date.strftime("%m")
    day = date.strftime("%d")
    year = date.strftime("%y")
    date = ('_'+month + '_' + day + '_' + year)

    parser = argparse.ArgumentParser(description='ABAQUS model generator')
    parser.add_argument('--directory', type=str,default='Subject Data')
    parser.add_argument('--sub', type=str, default='MTSFX06',help='pt id')
    parser.add_argument('--side', type=str, default='',help='L/R side only for Cadaver_Data')
    parser.add_argument('--angles', type=str, default='30,50,55,60,65,70', help='comma separated angles (degrees)')
    parser.add_argument('--loads', type=str, default='100', help='comma separated loads (N)')
    parser.add_argument('--flip', type=str, default='0,0,0,0,0,0', help='comma separated flip condiiton')
    parser.add_argument('--scale', type=float, default=0.001,help='scaling to meters')
    parser.add_argument('-t','--thresh_dist', type=float, default=0.02,
                        help='distance threshold for fixation load direction check.')
    parser.add_argument('--buffer', type=float, default=0.005, help='distance buffer for output nodes/elements (m)')
    args = parser.parse_args()
    
    args_dict = vars(args)
    old_direct = args.directory
    #axial loading at minimum, add angles listed here:
    angles = [float(i) for i in args.angles.split(',')]
    loads = [float(i) for i in args.loads.split(',')]
    
    # mimics outputs should be stored in an 'abaqus_files' folder
    # with the name f'{ID}_{SIDE}_MT{MTNO}'
    args.directory = ('Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/'+
                      args.directory)
    if 'Cadaver' in args.directory:
        directory = args.directory+f"/{args.sub}/{args.side}/abaqus_files/"
    else:
        directory = args.directory+f"/{args.sub}/abaqus_files/"
    
    with open(f'{directory}{date}args.txt','w') as file:
        for arg, value in args_dict.items():
            file.write(f'{arg}: {value}\n')
    
    files = os.listdir(directory)
    files = [file for file in files if '_R_' in file or '_L_' in file]
    files = [file for file in files if ('MT') in file]
    files = [file for file in files if 'new' not in file]
    
    #orientation for anisotropy
    orientation = ['*Orientation, name=Ori-1',
                   '1., 0., 0., 0., 1., 0.',
                   '1, 0.']
    
    #Read bodyweight from MTSFX Subject Demographics or assign loads
    if 'Cadaver' in args.directory:
        load = loads
    else:
        demo = pd.read_excel(args.directory+'/MTSFX Subject Demographics.xlsx',sheet_name='Demographics')
        demo = demo[demo['Participant ID']==args.sub]['BW_lbs'].iloc[0]
        
        load = (demo/2.2)*9.81*0.175 # lbs -> kg -> N -> N / Metatarsal
        load = [load * (ld/100) for ld in loads]
    
    for file_idx, file in enumerate(files):
        side = '_L_'
        if '_R_' in file:
            side = '_R_'
            
        filename, file_ext = os.path.splitext(file)
        
        base_file = directory+file
        a_new_file = directory+filename+'_axial_new'+file_ext
        b_new_file = directory+filename+'_bend_new'+file_ext
        
        nodes = read_inp(base_file, '*NODE','*')
        
        elements = read_inp(base_file,'*ELEMENT','*',True)
        surf_elements = read_inp(base_file,'*SURFACE','*',True)
        surf_nodes = read_inp(base_file,'*NSET, NSET=NS_Surface','*',True)
        element_sets = read_inp(base_file,'*ELSET','*MATERIAL',True)
        materials = read_inp(base_file,'*MATERIAL','*bleep',True)
        
        x = [float(node.split(',')[1]) for node in nodes]
        y = [float(node.split(',')[2]) for node in nodes]
        z = [float(node.split(',')[3]) for node in nodes]
        
        rangx = max(x) - min(x)
        rangy = max(y) - min(y)
        rangz = max(z) - min(z)
        
        largest = max(rangx,rangy,rangz)
        
        if rangz == largest:
            layout = [2,1,0]
        elif rangy == largest:
            layout = [1,2,0]
        else:
            layout = [0,1,2]
        if args.flip.split(',')[file_idx] == '1':
            a = layout[1]
            b = layout[2]
            layout = [layout[0],b,a]
        
        axis = layout[0]+1
        new_nodes = scale_n_PCA(nodes,args.scale,layout)
        
        x = [float(node.split(',')[1]) for node in new_nodes]
        y = [float(node.split(',')[2]) for node in new_nodes]
        z = [float(node.split(',')[3]) for node in new_nodes]
        new_coords = np.column_stack((x, y, z))
        
        min_z = min([float(node.split(',')[axis]) for node in new_nodes[2:]])
        max_z = max([float(node.split(',')[axis]) for node in new_nodes[2:]])
        # this only works for resliced mimics files
        # assumes load direction is positive along long axis
        # how can we determine if it's the other way?
        
        load_node = 1 #assume load node is at the top of the scan... for now
        
        bend_dir = layout[1]+1
        target = max_z - args.thresh_dist
        
        tan_30_deg = -1 / np.sqrt(3) 
        tolerance = 0.003
        
        check_nodes = [
            node.split(',')[0] for node in new_nodes[2:] if float(
                node.split(',')[axis]) < target and (
                    float(node.split(',')[axis]) > -target)]
        
        
        y_check = [float(nd.split(',')[bend_dir]) for nd in new_nodes[2:] 
                   if int(nd.split(',')[0] in check_nodes)]
        y_check = sum(y_check)
        
        bend_neg = y_check < 0
        
        bend_load_node = None
        bend_value = 0  
        
        # Loop through nodes and filter based on z-axis and bend direction conditions
        for ind, node in enumerate(new_coords):
            if float(node[axis-1]) > 0:
                continue # if z is positive, skip, this is the bottom of the bone.
                
            if bend_neg and float(node[bend_dir-1]) > bend_value:
                bend_value = float(node[bend_dir-1])
                bend_z = float(node[axis-1])
                bend_load_node = ind+1 #grab node farthest from the y axis
            if not bend_neg and float(node[bend_dir-1]) < bend_value:
                bend_value = float(node[bend_dir-1])
                bend_z = float(node[axis-1])
                bend_load_node = ind+1 #grab node farthest from the y axis
                
        fixed_nodes = [node.split(',')[0] for node in new_nodes[2:] if float(node.split(',')[axis]) > target]
        
        active_nodes = [
            node.split(',')[0] for node in new_nodes[2:] if float(
                node.split(',')[axis]) < target - args.buffer and (
                    float(node.split(',')[axis]) > min_z + args.buffer)]
                    
        active_nodes = [s.replace("\t", "") for s in active_nodes]
        
        active_nodes2 = [
            node.split(',')[0] for node in new_nodes[2:] if float(
                node.split(',')[axis]) < target - args.buffer and (
                    float(node.split(',')[axis]) > bend_z + args.buffer)]
                    
        active_nodes2 = [s.replace("\t", "") for s in active_nodes2]
    
        active_elements = [int(ele.split(',')[0]) for ele in elements[1:] if all(
            node.strip() in active_nodes for node in ele.split(',')[1:])]
        
        active_elements2 = [int(ele.split(',')[0]) for ele in elements[1:] if all(
            node.strip() in active_nodes2 for node in ele.split(',')[1:])]

        fixed_nodes = repack_nodes(fixed_nodes)
        fixed_nodes = '\n'.join(fixed_nodes)
        new_nodes.insert(0,'*NODE')
        new_nodes.insert(0,'*Part, name=PART-1')
        
        active_nodes2 = repack_nodes(active_nodes2)
        active_nodes2 = '\n'.join(active_nodes2)
        
        active_elements2 = repack_nodes(active_elements2)
        active_elements2 = '\n'.join(active_elements2)
        
        active_nodes = repack_nodes(active_nodes)
        active_nodes = '\n'.join(active_nodes)
        
        active_elements = repack_nodes(active_elements)
        active_elements = '\n'.join(active_elements)
        
        element_sets2 = element_sets.copy()
        element_sets.append('*ELSET, ELSET=Active_Elements')
        element_sets.append(active_elements)
        
        for i, line in enumerate(element_sets):
            if '*solid section,' in line.lower():
                edit = line.split(',')
                edit.insert(2, 'orientation=Ori-1')
                element_sets[i] = ','.join(edit)
        element_sets.append('*End Part')
        element_sets.append('*Assembly, name=Assembly')
        element_sets.append('*Instance, name=PART-1-1, part=PART-1')
        element_sets.append('*End Instance')
        
        element_sets2.append('*ELSET, ELSET=Active_Elements')
        element_sets2.append(active_elements2)
        
        for i, line in enumerate(element_sets2):
            if '*solid section,' in line.lower():
                edit = line.split(',')
                edit.insert(2, 'orientation=Ori-1')
                element_sets2[i] = ','.join(edit)
        element_sets2.append('*End Part')
        element_sets2.append('*Assembly, name=Assembly')
        element_sets2.append('*Instance, name=PART-1-1, part=PART-1')
        element_sets2.append('*End Instance')
        
        node_sets = ['*NSET, NSET=Load_Node, instance=PART-1-1',f'{load_node}',
                     '*NSET, NSET=Bend_Load_Node, instance=PART-1-1',f'{bend_load_node}',
                     '*NSET, NSET=Fixed_Nodes, instance=PART-1-1',fixed_nodes,
                     '*NSET, NSET=Active_Nodes, instance=Part-1-1',active_nodes]
        node_sets2 = ['*NSET, NSET=Load_Node, instance=PART-1-1',f'{load_node}',
                     '*NSET, NSET=Bend_Load_Node, instance=PART-1-1',f'{bend_load_node}',
                     '*NSET, NSET=Fixed_Nodes, instance=PART-1-1',fixed_nodes,
                     '*NSET, NSET=Active_Nodes, instance=Part-1-1',active_nodes2]
        
        constraint = [('*Coupling, constraint name=Constraint-1, ref node'+
                       '=Load_Node, surface=PART-1-1.SURFACE, influence radius=0.005'),
                      '*Kinematic','*End Assembly']
        constraint2 = [('*Coupling, constraint name=Constraint-1, ref node'+
                        '=Load_Node, surface=PART-1-1.SURFACE, influence radius=0.005'),
                       '*Kinematic','*End Assembly']

        for i, mat in enumerate(materials):
            if mat.lower().startswith('*material'):
                ram = mat.split('=')[-1].replace(' ', '')  # Remove spaces
                materials[i] = f'*MATERIAL, NAME={ram}'
                
            if mat.lower().startswith('*elastic'):
                mat_list = list(mat)
                mat_list.insert(11, ', type=ENGINEERING CONSTANTS')
                materials[i] = ''.join(mat_list)
                
                ni = i + 1
                emod = float(materials[ni].split(',')[0])
                
                if rangz == largest:
                    mat_vals = [0.574*emod,0.577*emod,emod, #E1-3
                                0.427,0.234,0.405, #v12,23,31
                                0.195*emod,0.216*emod,0.265*emod] #G13,31,23
                elif rangy == largest:
                    mat_vals = [0.574*emod,emod,0.577*emod, #E1-3
                                0.405,0.427,0.234, #v12,23,31
                                0.216*emod,0.195*emod,0.265*emod] #G13,31,23
                else:
                    mat_vals = [emod,0.577*emod,0.574*emod, #E1-3
                                0.405,0.234,0.427, #v12,23,31
                                0.265*emod,0.216*emod,0.195*emod] #G13,31,23
                        
                materials[ni] = f"{', '.join(map(str, mat_vals[:8]))},"
                materials.insert(ni+1,f'{mat_vals[8]}')
                
                
        boundary = ['*Boundary','Fixed_Nodes, 1, 1',
                    'Fixed_Nodes, 2, 2','Fixed_Nodes, 3, 3']
        
        if 'MT2' in file:
            mtno = 2
        elif 'MT3' in file:
            mtno = 3
        else:
            mtno = 4
        
        steps = [f'*Step, NAME = MT{mtno}{side}{abs(int(load[0]))}N_axial','*STATIC',
                 '*Cload',f'Load_Node, 3, {load[0]}','*Restart, write, frequency=0',
                 '*Output, field','*Node Output, nset=Active_Nodes','CF, RF, U',
                 '*Element Output, elset=PART-1-1.Active_Elements, directions=YES',
                 'E, EVOL, MISES, MISESMAX, S',
                 '*Output, history','*Contact Output','CSMAXSCRT, ','*End Step']
    
        for load_ in load:
            if load_ == load[0]:
                continue
            ram = [f'*Step, NAME = MT{mtno}{side}{abs(int(load_))}_N_axial',
                   '*STATIC',
                   '*Cload',f'Load_Node, 3, {load_}',
                   '*End Step']
            steps += ram
        
        
        new_data = (new_nodes + elements + surf_elements + orientation + 
                    element_sets + node_sets + constraint + materials +
                    boundary + steps)
        write_inp(new_data, a_new_file)        
        
        for angle in angles:
            if angle == angles[0]:
                for load_ in load:
                    z_force = np.cos(angle*np.pi/180) * load_
                    if bend_neg:
                        load_ = -load_
                    y_force = np.sin(angle*np.pi/180) * load_
                    
                    if load_ == load[0]:
                        steps = [f'*Step, NAME = MT{mtno}{side}{abs(int(load_))}N_{int(angle)}deg','*STATIC',
                                 '*Cload',f'Bend_Load_Node, {bend_dir}, {y_force}',
                                 f'Bend_Load_Node, 3, {z_force}',
                                 '*Restart, write, frequency=0',
                                 '*Output, field','*Node Output, nset=Active_Nodes','CF, RF, U',
                                 '*Element Output, elset=PART-1-1.Active_Elements, directions=YES',
                                 'E, EVOL, MISES, MISESMAX, S',
                                 '*Output, history','*Contact Output','CSMAXSCRT, ','*End Step']
                    else:
                        steps += [f'*Step, NAME = MT{mtno}{side}{abs(int(load_))}N_{int(angle)}deg',
                                  '*STATIC',
                                  '*Cload',f'Bend_Load_Node, {bend_dir}, {y_force}',
                                  f'Bend_Load_Node, 3, {z_force}',
                                  '*End Step']
            else:
                for load_ in load:
                    z_force = np.cos(angle*np.pi/180) * load_
                    if bend_neg:
                        load_ = -load_
                    y_force = np.sin(angle*np.pi/180) * load_
                    
                    ram = [f'*Step, NAME = MT{mtno}{side}{abs(int(load_))}N_{int(angle)}deg',
                           '*STATIC',
                           '*Cload',f'Bend_Load_Node, {bend_dir}, {y_force}',
                           f'Bend_Load_Node, 3, {z_force}',
                           '*End Step']
                    steps += ram
        
        new_data = (new_nodes + elements + surf_elements + orientation + 
                    element_sets2 + node_sets2 + constraint2 + materials +
                    boundary + steps)
        write_inp(new_data, b_new_file)
        
        
    
    input_dir = directory
    
    batch_script = f"""@echo off
    SET INPUT_DIR="{input_dir}"
    
    for %%f in (%INPUT_DIR%\\*_new.inp) do (
        for %%a in ("%%f") do (
            echo Running Abaqus for file: %%~na
            abaqus job=%%~na input="%%f"
        )
    )
    """
    
    # Write the batch script to a file
    with open(directory + "run_abq_jobs.bat", "w") as file:
        file.write(batch_script)
    
    print("Batch script generated successfully.")