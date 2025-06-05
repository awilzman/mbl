# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:51:47 2024

@author: Andrew
"""
import pandas as pd
import numpy as np
import argparse
import os
import h5py
import open3d as o3d

def preprocess_point_cloud(pc, dims=[300, 100, 100]):
    # Voxelization
    edge_length_x = pc['x'].max() - pc['x'].min()
    edge_length_y = pc['y'].max() - pc['y'].min()
    edge_length_z = pc['z'].max() - pc['z'].min()

    # Calculate the maximum edge length to determine the length scale factor
    max_edge_length = max(edge_length_x, edge_length_y, edge_length_z)

    # Use the maximum edge length as the length scale factor
    length_scale_factor = max_edge_length / max(dims)

    # Use this length scale factor for voxelization
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=length_scale_factor)
    voxel_centers = np.asarray(voxel_grid.get_voxels()).mean(axis=1)

    # Resampling to target dimensions using NumPy
    voxel_centers_resampled = np.empty(dims + (3,))
    for i in range(3):
        voxel_centers_resampled[:, :, :, i] = np.linspace(
            voxel_centers[:, i].min(), voxel_centers[:, i].max(), dims[i])

    # Creating point cloud with length scale factor as a feature
    length_scale_feature = np.full(dims + [1], length_scale_factor)
    voxel_centers_resampled_with_length_scale = np.concatenate([voxel_centers_resampled, length_scale_feature], axis=3)

    return voxel_centers_resampled_with_length_scale,length_scale_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str,default='Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/')
    parser.add_argument('-samp', type=str,default='')
    args = parser.parse_args()
    
    directory = args.d+'Data/'

    samps = os.listdir(directory+'Compressed/')
    
    if args.samp != '':
        if args.samp not in samps:
            print('Sample not available')
            # look for nifti, if no, look for dicoms, if no, return error
            # if one of those exist, just turn on flags to reseg/reclou as needed
        else:
            samps = args.samp
    for s in samps:
        MTs = os.listdir(f'{directory}Compressed/{s}')
        for MT in MTs:
            try:
                with h5py.File(f'{directory}Compressed/{s}/{MT}', 'r') as hf:
                    bone = pd.DataFrame(hf['Pointcloud'][:], columns=['x','y','z','d'])
                    MTno = hf['MTno']
                    side = hf['Side']
            except:
                print(f'unable to load {s}')
            