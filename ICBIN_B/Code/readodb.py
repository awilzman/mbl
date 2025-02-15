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

# Idea: 
#     Analyze the failed volume and respective location within the bone 
#     to measure the failure probability as a function over cycles.
#     That is to say, I will have a measure F(x,u) = K*sum(f_n(Load_n,rate_n,f_{n-1}))
#     where n is the number of loading cycles, and we sum the damage which is 
#     a function of the load, loading rate, and previous damage measure. F(x,u) 
#     is then the probability of failure after n loading cycles.

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
                'Tsai Wu Strained Volume': entry['tw_str_vol'],
                'von Mises Stressed Volume': entry['vm_str_vol'],
                'Max Tsai Wu Ratio': entry['tw_md_vr'],
                'Max von Mises Ratio': entry['vm_md_vr'],
                'Max Displacement': entry['max_displ']}
            writer.writerow(row)

    # Tsai-Wu failure criteria parameters 
    #https://www.sciencedirect.com/science/article/pii/S1751616111001445
    #https://www.sciencedirect.com/science/article/pii/S8756328214003664
    
    #Kemper A, McNally C, Kennedy E, Manoogian S, Duma S. 
    # The material properties of human tibia cortical bone in tension 
    # and compression: implications for the tibia index. InProceedings of the 
    # 20th Enhanced Safety of Vehicles Conference, Lyon, France 
    # 2007 Jun 18 (pp. 07-0470).
    
def process_files(directory, files, output_file, header):
    combined_data = []

    # Tsai-Wu failure criteria parameters
    X_c, X_t = -0.01, 0.006
    Y_c, Y_t = -0.006, 0.003
    S = 0.0009

    F_1 = (1/X_t) + (1/X_c)
    F_2 = (1/Y_t) + (1/Y_c)
    F_4 = -1/(X_t * X_c)
    F_5 = -1/(Y_t * Y_c)
    F_6 = S**(-2)
    F_7 = 1/(2*Y_t) * (1 - (F_1 + F_2) * Y_t - Y_t**2 * (F_4 + F_5))

    vm_yield_stress = 16e6  # von Mises yield stress threshold (16 MPa)

    for file_name in files:
        odb = openOdb(os.path.join(directory, file_name))
        element_centroids = {}
        odb_instance = odb.rootAssembly.instances['PART-1-1']
        node_coords = {node.label: node.coordinates for node in odb_instance.nodes}
        
        for elem in odb_instance.elements:
            elem_nodes = elem.connectivity  
            centroid = np.mean([node_coords[nid] for nid in elem_nodes], axis=0)
            element_centroids[elem.label] = centroid  

        for step_key, step in odb.steps.items():
            for frame_index, frame in enumerate(step.frames):
                if frame_index == 0:
                    continue  

                counted_elements_tsai_wu = set()
                counted_elements_vm = set()
                max_vm_stress = 0
                str_vol_step = 0
                vm_vol_step = 0

                evols = {evol.elementLabel: evol.data * 1e9 for evol in frame.fieldOutputs['EVOL'].values}  
                disps = {dcp.nodeLabel: np.linalg.norm(dcp.data) for dcp in frame.fieldOutputs['U'].values}
                max_displ = max(disps.values()) * 1e3 if disps else 0  

                # Initialize slice_total_volumes with all elements in evols
                slice_total_volumes = {}
                for elem_id, volume in evols.items():
                    centroid = element_centroids.get(elem_id)
                    if centroid is None:
                        continue  # Skip if element centroid not found (unlikely)
                    z_coord = centroid[2]
                    slice_idx = int(z_coord / 0.01)
                    if slice_idx not in slice_total_volumes:
                        slice_total_volumes[slice_idx] = 0.0
                    slice_total_volumes[slice_idx] += volume

                slice_damaged_volumes = {}
                slice_vm_damaged_volumes = {}
                prp_vals = []

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
                    principal_strains = np.linalg.eigvalsh(strain_tensor)  # Eigenvalues are sorted automatically
                
                    # Assign p_max and p_min
                    prp_vals.append([principal_strains[-1],principal_strains[0]])
                    
                    e2 = e22 if abs(e22) > abs(e33) else e33
                    shr = max(abs(e12), abs(e13), abs(e23))

                    tsai_wu = (F_1 * e11 + F_2 * e22 + F_4 * (e11**2) + 
                               F_5 * (e2**2) + F_6 * (shr**2) + 2 * F_7 * e11 * e2)

                    element_volume = evols[elem_id]
                    z_coord = element_centroids[elem_id][2]
                    slice_idx = int(z_coord / 0.01)

                    if tsai_wu >= 1 and elem_id not in counted_elements_tsai_wu:
                        if slice_idx not in slice_damaged_volumes:
                            slice_damaged_volumes[slice_idx] = 0.0
                        slice_damaged_volumes[slice_idx] += element_volume
                        counted_elements_tsai_wu.add(elem_id)

                tw_max_damage_ratio = (
                    max(
                        (slice_damaged_volumes[slice_idx] / slice_total_volumes[slice_idx]  
                         for slice_idx in slice_damaged_volumes if slice_idx in slice_total_volumes)
                    ) if slice_damaged_volumes else 0
                )

                for vm in frame.fieldOutputs['MISESMAX'].values:
                    elem_id = vm.elementLabel
                    vm_value = vm.data

                    if vm_value > max_vm_stress:
                        max_vm_stress = vm_value

                    if vm_value > vm_yield_stress and elem_id not in counted_elements_vm and elem_id in evols:
                        z_coord = element_centroids[elem_id][2]
                        slice_idx = int(z_coord / 0.01)

                        if slice_idx not in slice_vm_damaged_volumes:
                            slice_vm_damaged_volumes[slice_idx] = 0.0
                        slice_vm_damaged_volumes[slice_idx] += evols[elem_id]
                        counted_elements_vm.add(elem_id)

                vm_max_damage_ratio = (
                    max(
                        (slice_vm_damaged_volumes[slice_idx] / slice_total_volumes[slice_idx]  
                         for slice_idx in slice_vm_damaged_volumes if slice_idx in slice_total_volumes)
                    ) if slice_vm_damaged_volumes else 0
                )
                
                tw_str_vol = 0.0
                for v in slice_damaged_volumes.values():
                    tw_str_vol += v  # Ensure all values are numeric
                
                vm_str_vol = 0.0
                for v in slice_vm_damaged_volumes.values():
                    vm_str_vol += v  # Same as above

                
                combined_data.append({
                    'step_key': step_key,
                    'prp_max': max(max(prp_vals)),
                    'prp_min': min(min(prp_vals)),
                    'max_vm_stress': max_vm_stress,
                    'tw_str_vol': tw_str_vol,
                    'vm_str_vol': vm_str_vol,
                    'tw_md_vr': tw_max_damage_ratio,
                    'vm_md_vr': vm_max_damage_ratio,
                    'max_displ': max_displ
                })

        odb.close()

    write_data_to_csv(output_file, header, combined_data)

if __name__ == "__main__":
    date = datetime.datetime.now()
    month = date.strftime("%m")
    day = date.strftime("%d")
    year = date.strftime("%y")
    date = ('_'+month + '_' + day + '_' + year)

    parser = argparse.ArgumentParser(description='ABAQUS odb reader')
    parser.add_argument('--directory', type=str,default='Cadaver_Data')
    parser.add_argument('--sub', type=str, default='2205033M', help='pt id')
    parser.add_argument('--side', type=str, default='L',help='L/R side only for Cadaver_Data')
    args = parser.parse_args()
    args.directory = ('Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/'+
                      args.directory+'/')
    if args.side == '':
        directory = args.directory + args.sub + '/abaqus_files/'
    else:
        directory = args.directory + args.sub + '/' + args.side + '/abaqus_files/'

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
        'Tsai Wu Strained Volume',
        'von Mises Stressed Volume',
        'Max Tsai Wu Ratio',
        'Max von Mises Ratio',
        'Max Displacement'
    ]
    process_files(directory,files,output_file,header)

