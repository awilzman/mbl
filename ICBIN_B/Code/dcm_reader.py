# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:23:05 2023

@author: arwilzman
"""

#%% Initialize
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from sklearn.decomposition import IncrementalPCA
import hdbscan
import dicom2nifti
import h5py
import nibabel as nib

def download_dicoms(directory,sample,meas,file_name,file_url):
    for i in range(500):
        num=str(i).zfill(5)
        file_path_ram = file_name+'_'+num+'.DCM'
        file_url_ram = file_url+sample+'/'+meas+'/'+file_path_ram
        response = requests.get(file_url_ram)
        file_content = response.content
        if response.status_code == 200:
            local_file_path = directory+file_path_ram
            with open(local_file_path, 'wb') as file:
                file.write(file_content)
        else:
            print(f"Done at {file_path_ram} using {file_url_ram}")
            break

if __name__ == "__main__":
    # Specify the URL of the file you want to download
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='C:/Users/arwilzman/Documents/')
    parser.add_argument('-samp', type=str, default='00000218') #218
    parser.add_argument('-meas', type=str, default='00002208') #2208, 2768
    parser.add_argument('-file', type=str, default='RAW1') #C0002458, C0003004
    parser.add_argument('-name', type=str, default='MARA14-RAW1') #MARA14-1,2
    parser.add_argument('-threshold', type=float, default=2,help='initial threshold')
    parser.add_argument('-trunc', type=float, default=10,help='pointcloud truncation factor')
    parser.add_argument('-scale', type=float, default=0.001,help='scale factor')
    parser.add_argument('-res', type=float, default=82,help='resolution, microns')
    parser.add_argument('-c_slope', type=float, default=0.000357,help='calibration slope')
    parser.add_argument('-c_int', type=float, default=-0.0012625,help='calibration intercept')
    parser.add_argument('-grow_tol', type=float, default=0.001,help='region grow tolerance (m)')
    parser.add_argument('-grow_iter', type=int, default=10)
    parser.add_argument('-resave',type=bool, default=False)
    args = parser.parse_args()
    
    file_url = 'http://xct.wpi.edu/DISK2/MICROCT/DATA/'
    
    plot = False # for data visualization
    elev = 25
    azim = range(0,360,360)
    
    directory = (args.d+args.name)
    DL=False
    GTG=False
    
    if not os.path.exists(directory+'.h5'):
        args.resave = True
        print('Resaving because h5 does not exist')
        if not os.path.exists(directory+'.nii.gz'):
            if not os.path.exists(directory+'/'):
                print('I will need to download')
                DL = True
            else:
                print('Compressing DICOMs')
        else:
            print('Reading .nii.gz')      
        
    elif args.resave:
        print('Recompressing .h5')
    if not args.resave and not DL:
        print('Good to go.')
        GTG = True
            
    
    
    #%%
    if DL:
        if not os.path.exists(directory+'/'):
            os.makedirs(directory+'/')
            download_dicoms(directory+'/',args.samp,args.meas,args.file,file_url)
        if not os.path.exists(directory+'-crtxseg/'):
            os.makedirs(directory+'-crtxseg/')
            download_dicoms(directory+'-crtxseg/',args.samp,args.meas,args.file+'_CRTXSEG_1',file_url)
        if not os.path.exists(directory+'-blck/'):
            os.makedirs(directory+'-blck/')
            download_dicoms(directory+'-blck/',args.samp,args.meas,args.file+'_BLCK_1',file_url)
        
    #%%    
    if GTG:
        with h5py.File(f'{directory}.h5', 'r') as hf:
            bone = pd.DataFrame(hf['seg'][:], columns=['x','y','z','d','c'])
            output = pd.DataFrame(hf['raw'][:], columns=['x','y','z','d'])
            crtx = pd.DataFrame(hf['crtx'][:], columns=['x','y','z','d'])
            blck = pd.DataFrame(hf['blck'][:], columns=['x','y','z','d'])
        print('Loaded from h5')
    else:
        # compress raw dicoms to .nii.gz
        
        ni_file = directory+'.nii.gz'
        if not os.path.exists(ni_file):
            dicom2nifti.dicom_series_to_nifti(directory,ni_file,reorient_nifti=True)
        
        image = nib.load(ni_file)
        # Get the voxel data from the NIfTI file
        data = image.get_fdata()
        
        if data.max()*args.c_slope+args.c_int < args.threshold:
            args.threshold = (data.max()*args.c_slope+args.c_int)*0.2
        # Get the voxel coordinates, threshold
        coords = np.array(np.where((data*args.c_slope+args.c_int)>args.threshold)).T
        # Create a Pandas DataFrame with columns ['x', 'y', 'z', 'd']
        output = pd.DataFrame(coords*args.scale*args.res,columns=['x', 'y', 'z'])[::args.trunc]
        # Add a column 'd' with the voxel values
        output['d'] = data[coords[:,0],coords[:,1],coords[:,2]][::args.trunc]
        output['d'] = output['d']*args.c_slope + args.c_int
        
        print('Clustering')
        hdb = hdbscan.HDBSCAN(min_cluster_size=1000)
        output['cluster'] = hdb.fit_predict(output[['x', 'y', 'z']])
        output.to_pickle(args.d+args.name+'.pkl')
        print('Cluster completed')
        # Compute cluster sizes
        cluster_sizes = output['cluster'].value_counts()
        cluster_sizes = cluster_sizes[cluster_sizes.index>=0]
        # Identify the cluster label with the largest & second largest size
        largest_cluster_label = cluster_sizes.idxmax()
        lesser_cluster_label = cluster_sizes[cluster_sizes.index!=largest_cluster_label].idxmax()
        bone = output[output['cluster']==largest_cluster_label].reset_index(drop=True)
        lateral = output[output['cluster']==lesser_cluster_label].reset_index(drop=True)
        lateral = lateral.iloc[0]
        # find anterior and lateral point on bone
        # anterior assumed to be in the direction of highest y value of the scan
        # lateral assumed to be in the direction of the next biggest cluster on x axis 
        if lateral['x'] > bone.loc[0,'x']:
            bone_lateral_point = bone['x'].idxmax()
        else:
            bone_lateral_point = bone['x'].idxmin()
        bone.loc[bone_lateral_point,'cluster'] = -1 #identify lateral point
        bone_anterior_point = bone['y'].idxmax()
    
        # PCA
        print('Applying PCA')
        ipca = IncrementalPCA(n_components=3, batch_size=1000)
        original_coordinates=bone[['x', 'y', 'z']].values
        ipca.fit(original_coordinates)
        # Transform the original coordinates using the fitted PCA
        reoriented_coordinates = ipca.transform(original_coordinates)
        # Calculate the rotation matrix to align principal axes to z, x, y
        rotation_matrix = ipca.components_  # The columns are the principal components
        bone.loc[:,['x','y','z']]=reoriented_coordinates
                
        #get crtx data and rotate
        ni_file = directory+'-crtxseg.nii.gz'
        if not os.path.exists(ni_file):
            dicom2nifti.dicom_series_to_nifti(directory+'-crtxseg',ni_file,reorient_nifti=True)
        image = nib.load(ni_file)
        # Get the voxel data from the NIfTI file
        data = image.get_fdata()
        coords = np.array(np.where(data>0)).T
        crtx = pd.DataFrame(coords,columns=['x', 'y', 'z'])[::args.trunc]
        # Add a column 'd' with ones as values
        crtx['d'] = np.ones(len(crtx))
        # Transform the original coordinates using the fitted PCA
        original_coordinates=crtx[['x', 'y', 'z']].values
        reoriented_coordinates = ipca.transform(original_coordinates)
        # Calculate the rotation matrix to align principal axes to z, x, y
        rotation_matrix = ipca.components_  # The columns are the principal components
        crtx.loc[:,['x','y','z']]=reoriented_coordinates
        
        #get blck data
        ni_file = directory+'-blck.nii.gz'
        if not os.path.exists(ni_file):
            dicom2nifti.dicom_series_to_nifti(directory+'-blck',ni_file,reorient_nifti=True)
        image = nib.load(ni_file)
        # Get the voxel data from the NIfTI file
        data = image.get_fdata()
        coords = np.array(np.where(data>0)).T
        blck = pd.DataFrame(coords,columns=['x', 'y', 'z'])[::args.trunc]
        # Add a column 'd' with ones as values
        blck['d'] = np.ones(len(blck))
        # Transform the original coordinates using the fitted PCA
        original_coordinates=blck[['x', 'y', 'z']].values
        reoriented_coordinates = ipca.transform(original_coordinates)
        # Calculate the rotation matrix to align principal axes to z, x, y
        rotation_matrix = ipca.components_  # The columns are the principal components
        blck.loc[:,['x','y','z']]=reoriented_coordinates
        
        if bone.loc[bone_anterior_point,'y'] != bone['y'].max(): 
            # rerotate anterior point is max y
            # remember lateral point is marked by cluster = -1 for later plot labeling
            rotation_angle = np.arccos(np.clip(np.dot(rotation_matrix[2], [0, 0, 1]), -1.0, 1.0))
            rotation_matrix_z = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                          [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                          [0, 0, 1]])
            bone[['x', 'y', 'z']] = np.dot(bone.loc[:,['x','y','z']], rotation_matrix_z.T)
            crtx[['x', 'y', 'z']] = np.dot(crtx.loc[:,['x','y','z']], rotation_matrix_z.T)
            blck[['x', 'y', 'z']] = np.dot(blck.loc[:,['x','y','z']], rotation_matrix_z.T)
        output = output.drop(columns=['cluster'])
            
        # Create binary save files
        with h5py.File(f'{args.d}{args.name}.h5', 'w') as hf:
            hf.create_dataset('raw', data=output)
            hf.create_dataset('seg', data=bone)
            hf.create_dataset('crtx', data=crtx)
            hf.create_dataset('blck', data=blck)
            
        # Create .txt file compatible with 3matic
        bone[['x','y','z']].round(3).to_csv(args.d+args.name+'.txt',
                                              index=False,header=False,sep='\t',
                                              float_format='%.3f')    
        
    
    #%% Visualize the results
    if plot:
        for a in azim:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(bone['x'], bone['y'], bone['z'], c=bone['d'], cmap='bone', marker='o', alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(args.name)
            ax.view_init(elev,a)
            plt.show()
    #%%

