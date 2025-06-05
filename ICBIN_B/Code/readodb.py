# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:15:12 2023

@author: arwilzman
"""
from odbAccess import *
import numpy as np
import datetime
import os
import argparse
import csv

def write_data_to_csv(file, header, data):
    output_file = file.replace('.odb', '_data.csv')
    with open(output_file, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for entry in data:
            row = {
                'Step Key': entry['step_key'],
                'Max Principal Strain': entry['prp_max'],
                'Min Principal Strain': entry['prp_min'],
                'Max von Mises Stress': entry['max_vm_stress'],
                'Tsai-Wu Strained Volume': entry['tw_str_vol'],
                'von Mises Stressed Volume': entry['vm_o_vol'],
                'von Mises Strained Volume': entry['vm_e_vol'],
                'Total Volume': entry['tot_vol'],
                'Max Displacement': entry['max_displ']}
            writer.writerow(row)


def process_files(directory, files, output_file, header, lhs_samples):    
    pert_files = ['perturbed/' + file for file in os.listdir(os.path.join(directory, 'perturbed')) if file.endswith('.odb')]
    
    all_data = []
    
    for file_name in files:
        print(file_name)
        
        pre1 = '_'.join(file_name.split('_')[:3])
        pre2 = file_name.split('_')[3]
        relev_perts = [p for p in pert_files if ((pre1 in p) and (pre2 in p))]
        
        file_path = os.path.join(directory, file_name)
        c = process_odb(file_path,lhs_samples)
        all_data.extend(c)
        
        for pert in relev_perts:
            pert_path = os.path.join(directory, pert)
            print(pert)
            cp = process_odb(pert_path,lhs_samples)
            all_data.extend(cp)
    
    write_data_to_csv(output_file, header, all_data)

def process_odb(file_path,lhs_samples):
    combined_data = []
    
    tw_num_exp = 1
    vm_num_exp = 1
    #this needs to match everywhere when experimenting!
    # tw_num_exp = 64
    # vm_num_exp = 10
    # vm_o_min = 20e6
    # vm_o_max = 30e6
    # vm_e_min = 1e-4
    # vm_e_max = 2e-3
    
    X_c_samples = -0.0140 #np.array(lhs_samples[:, 0]) # fiber direction compression
    Y_c_samples = -0.0069 #np.array(lhs_samples[:, 1]) # transverse direction compression
    X_t_samples = 0.0048 #np.array(lhs_samples[:, 2]) # tension
    Y_t_samples = 0.0013 #np.array(lhs_samples[:, 3])
    S_samples = 0.0068 #np.array(lhs_samples[:, 4])
    
    # Compute response variables
    F_1 = (1 / X_t_samples) + (1 / X_c_samples)
    F_2 = (1 / Y_t_samples) + (1 / Y_c_samples)
    F_4 = -1 / (X_t_samples * X_c_samples)
    F_5 = -1 / (Y_t_samples * Y_c_samples)
    F_6 = S_samples ** (-2)
    F_7 = -0.2 * (F_4*F_5)**0.5

    vm_o_yield = 25e6 #np.arange(vm_o_min,vm_o_max,(vm_o_max-vm_o_min)/vm_num_exp)
    vm_e_yield = 0.0014 #np.arange(vm_e_min,vm_e_max,(vm_e_max-vm_e_min)/vm_num_exp)
    
    try:
        odb = openOdb(file_path,readOnly=False)
        res = True
    except:
        odb = openOdb(file_path)
        print('not resaving odb')
        res = False
    element_centroids = {}
    
    filename = os.path.basename(file_path)

    try:
        odb_instance = odb.rootAssembly.instances['PART-1-1']
    except:
        print('oof ' + file_name)

    node_coords = {node.label: node.coordinates for node in odb_instance.nodes}

    # Get centroids for elements
    for elem in odb_instance.elements:
        elem_nodes = elem.connectivity
        centroid = np.mean([node_coords[nid] for nid in elem_nodes], axis=0)
        element_centroids[elem.label] = centroid
        
    tw_strain_content = []
    tw_elems = {}
    vm_o_elems = {}
    vm_e_elems = {}
    
    # Deep Learning Targets
    dlt = {}
    for step_key, step in odb.steps.items():
        for frame_index, frame in enumerate(step.frames):
            if frame_index == 0:
                continue  # Skip the first frame
                
            max_vm_stress = 0
            evols = {evol.elementLabel: evol.data * 1e9 for evol in frame.fieldOutputs['EVOL'].values}
            
            tot_vol = 0
            for key, value in evols.items():
                tot_vol += value

            disps = {dcp.nodeLabel: np.linalg.norm(dcp.data) for dcp in frame.fieldOutputs['U'].values}  
            max_displ = max(disps.values()) * 1e3 if disps else 0

            # Initialize damage volume arrays for each experiment
            total_tsai_wu_volumes = np.zeros(tw_num_exp)
            total_vm_o_volumes = np.zeros(vm_num_exp)
            total_vm_e_volumes = np.zeros(vm_num_exp)

            # Tsai-Wu damage calculation
            prp_vals = []
            counted_elements_tsai_wu = {i: {} for i in range(tw_num_exp)} 
            counted_elements_e_vm = {i: {} for i in range(vm_num_exp)} 
            
            for strain in frame.fieldOutputs['E'].values:
                elem_id = strain.elementLabel
                if elem_id not in evols:
                    continue

                e11, e22, e33, e12, e13, e23 = strain.data

                strain_tensor = np.array([
                    [e11, e12, e13],
                    [e12, e22, e23],
                    [e13, e23, e33]
                ])
                
                # Compute principal strains (eigenvalues)
                principal_strains = np.linalg.eigvalsh(strain_tensor)
                
                # Assign p_max and p_min
                
                p_max = max(principal_strains[-1], principal_strains[0])
                p_min = min(principal_strains[-1], principal_strains[0])
                prp_vals.append([p_max,p_min])

                e2 = e22 if abs(e22) > abs(e11) else e11
                shr = max(abs(e12), abs(e13), abs(e23))

                tsai_wu = (F_1 * e33 + F_2 * e2 + F_4 * (e33**2) + 
                           F_5 * (e2**2) + F_6 * (shr**2) + F_7 * e33 * e2)
                segments = step_key.split('_')
                load = segments[2]
                tw_strain_content.append([load,e33,e2,shr])
                
                element_volume = evols[elem_id]
                
                #pressure modified
                v = 0.4
                k = 1.2
                J1 = e11 + e22 + e33
                J2 = (1/3)*((e11**2)+(e22**2)+(e33**2)-(e11*e22)-(e22*e33)-(e33*e11))+(e12**2)+(e23**2)+(e13**2)
                von_mises = ((k-1)/(2*k*(1-2*v)))*J1+(1/(2*k))*((
                    (J1*(k-1)/(1-(2*v)))**2+(12*k/((1+v)**2))*J2)**0.5)
                
                #for idx, tw in enumerate(tsai_wu): # also change 0s to idx if testing
                tw = tsai_wu
                if tw >= 1:
                    if elem_id not in counted_elements_tsai_wu[0]:
                        tw_elems[elem_id] = 1
                        counted_elements_tsai_wu[0][elem_id] = {"volume": element_volume}
                        total_tsai_wu_volumes[0] += element_volume
                elif elem_id not in counted_elements_tsai_wu[0]:
                    tw_elems[elem_id] = 0
                        
                #for idx, vm in enumerate(vm_e_yield):
                vm = vm_e_yield
                if von_mises >= vm:
                    if elem_id not in counted_elements_e_vm[0]:
                        vm_e_elems[elem_id] = 1
                        counted_elements_e_vm[0][elem_id] = {"volume": evols[elem_id]}
                        total_vm_e_volumes[0] += evols[elem_id]
                elif elem_id not in counted_elements_e_vm[0]:
                    vm_e_elems[elem_id] = 0
                
                dlt_key = step_key + '_' + str(elem_id)
                
                dlt[dlt_key] = {
                    "max_prp": p_max,
                    "min_prp": p_min,
                    "tsai_wu": tsai_wu,
                    "vm_strain": von_mises,
                    "volume": element_volume
                }
                
            # Von Mises damage calculation
            counted_elements_o_vm = {i: {} for i in range(vm_num_exp)}

            for vm in frame.fieldOutputs['MISESMAX'].values:
                elem_id = vm.elementLabel
                dlt_key = step_key + '_' + str(elem_id)
                vm_value = vm.data

                if vm_value > max_vm_stress:
                    max_vm_stress = vm_value

                #for idx, vm in enumerate(vm_o_yield):
                vm = vm_o_yield
                if vm_value >= vm:
                    if elem_id not in counted_elements_o_vm[0]:
                        vm_o_elems[elem_id] = 1
                        counted_elements_o_vm[0][elem_id] = {"volume": evols[elem_id]}
                        total_vm_o_volumes[0] += evols[elem_id]
                elif elem_id not in counted_elements_o_vm[0]:
                    vm_o_elems[elem_id] = 0
                    
                if dlt_key in dlt:
                    dlt[dlt_key]["vm_stress"] = vm_value
                else:
                    print('woah what?!')
                    dlt[dlt_key] = {"vm_stress": vm_value}
                    
            cd_key = filename.replace('.odb','') + '_' + load
            # Store results, switch between join function and single output 
            # depending on training vs evaluating
            combined_data.append({
                'step_key': cd_key,
                'prp_max': max(max(prp_vals)),
                'prp_min': min(min(prp_vals)),
                'max_vm_stress': max_vm_stress,
                'tw_str_vol': ', '.join(map(str, total_tsai_wu_volumes)), #total_tsai_wu_volumes[0],
                'vm_o_vol': ', '.join(map(str, total_vm_o_volumes)), #total_vm_o_volumes[0],
                'vm_e_vol': ', '.join(map(str, total_vm_e_volumes)), #total_vm_e_volumes[0],
                'tot_vol': tot_vol,
                'max_displ': max_displ
            })
            # add damaged volume binary parameters ONLY IF DONE TESTING
            if "Tsai-Wu" not in frame.fieldOutputs:
                    
                tw_fo = frame.FieldOutput(name="Tsai-Wu",description="Tsai-Wu Failure Condition",type=SCALAR)
                vm_o_fo = frame.FieldOutput(name="vMises Stress",description="von Mises Stresss Failure Condition",type=SCALAR)
                vm_e_fo = frame.FieldOutput(name="vMises Strain",description="von Mises Strain Failure Condition",type=SCALAR)
                
                tw_fo.addData(position=WHOLE_ELEMENT, 
                              instance=odb_instance,
                              labels=list(tw_elems.keys()), data=[(v,) for v in tw_elems.values()])
                vm_o_fo.addData(position=WHOLE_ELEMENT, 
                                instance=odb_instance,
                                labels=list(vm_o_elems.keys()), data=[(v,) for v in vm_o_elems.values()])
                vm_e_fo.addData(position=WHOLE_ELEMENT, 
                                instance=odb_instance,
                                labels=list(vm_e_elems.keys()), data=[(v,) for v in vm_e_elems.values()])
            
    odb.close()
    
    # Read existing data into a dictionary
    existing_data = {}
    output_f2 = file_path.replace('.odb', '_data.csv')
    
    if os.path.exists(output_f2):
        try:
            with open(output_f2, 'rb') as csvfile:  # Use 'rb' mode for Python 2
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_data[row['step']] = row
        except:
            existing_data = {}
    
    # Update existing data with new data
    for key, da in dlt.items():
        if key in existing_data:
            existing_data[key].update(da)
        else:
            new_entry = {'step': key}
            new_entry.update(da)
            existing_data[key] = new_entry
    
    # Write all data back to the CSV file
    with open(output_f2, 'wb') as csvfile:  # Use 'wb' mode for python2
        fieldnames = ['step', 'max_prp', 'min_prp', 'tsai_wu', 'vm_strain', 'volume', 'vm_stress']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_data.values():
            writer.writerow(row)
    
    return combined_data

if __name__ == "__main__":
    date = datetime.datetime.now()
    month = date.strftime("%m")
    day = date.strftime("%d")
    year = date.strftime("%y")
    date = ('_'+month + '_' + day + '_' + year)

    parser = argparse.ArgumentParser(description='ABAQUS odb reader')
    parser.add_argument('--directory', type=str,default='Cadaver_Data')
    parser.add_argument('--sub', type=str, default='', help='pt id')
    parser.add_argument('--side', type=str, default='',help='L/R side only for Cadaver_Data')
    args = parser.parse_args()
    args.directory = ('Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/'+
                      args.directory+'/')
    
    csv_file = args.directory + 'LHS.csv'
    directory = args.directory + args.sub + '/' + args.side + '/abaqus_files/'
    lhs_samples = []
    
    
    # use when testing
    # with open(csv_file) as file:
    #     csv_reader = csv.reader(file)
    #     header = next(csv_reader)  # Skip the header row
    #     for row in csv_reader:
    #         lhs_samples.append([float(val) for val in row])
    # lhs_samples = np.array(lhs_samples)
    #
    # tested values
    lhs_samples = np.array([[-0.014, -0.0069,  0.0048,  0.0013,  0.0068]])
    
    
    if args.side == '':
        directory = args.directory + args.sub + '/abaqus_files/'
        lhs_samples = lhs_samples[0]
        
    files = os.listdir(directory)
    files = [file for file in files if '_r_' in file.lower() or '_l_' in file.lower()]
    files = [file for file in files if 'mt' in file.lower()]
    files = [file for file in files if '.odb' in file]

    output_file = directory+args.sub+'_fe_data.csv'
    header = [
        'Step Key', 
        'Max Principal Strain',
        'Min Principal Strain',
        'Max von Mises Stress',
        'Tsai-Wu Strained Volume',
        'von Mises Stressed Volume',
        'von Mises Strained Volume',
        'Total Volume',
        'Max Displacement'
    ]
    
    process_files(directory,files,output_file,header,lhs_samples)

