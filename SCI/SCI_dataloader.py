# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# functions:
# load_qct(directory,studies)
#    purpose: load QCT data, see 
#       eg. Z:\_Current IRB Approved Studies\IRB_Closed\
#           FES_Rowing_SecondaryAnalysis\Final QCT Results
# load_masks(directory,com_name,ids,mat_shape,res,high_res=4,z_reduction=0.2)    
#    purpose: load masks, originally in pointcloud as output from segmentation,
#                into matrix space
# =============================================================================
import pandas as pd
import numpy as np
import os

def mask_registration(file,target_res,i,drop_factor):
    # Requires the following inputs:
    # file: directory to density point cloud of segmented bone
    # target_res: target resolution mm / element
    # i: sample number
    dc = pd.read_csv(file, sep=',', header=None) # density cloud
    dc = dc.sort_values(by=[2,0,1]) # sort by z, x, then y
    dc = dc.iloc[::2] # immediately downsample by 2
    dc = dc.reset_index(drop=True)
    if drop_factor < 1:
        drop_factor = drop_factor/2
        high_res_del = np.arange(int(round(dc.shape[0]*drop_factor))-1,1+int(
            round(dc.shape[0]-(dc.shape[0]*drop_factor))))
        dc = dc.drop(index=high_res_del)
        # this code will take a snippet of the beginning and end of the mask
        # used for higher resolution data
    dc = dc.reset_index(drop=True)
    # normalize min to 0 on all planes
    dc[0] = dc[0] - min(dc[0])
    dc[1] = dc[1] - min(dc[1])
    dc[2] = dc[2] - dc.iloc[0,2]
    # map indices for matrix
    x_ind = ((dc[0]/target_res).astype(int))
    y_ind = ((dc[1]/target_res).astype(int))
    z_ind = ((dc[2]/target_res).astype(int))
    x_ind, _ = pd.factorize(x_ind)
    y_ind, _ = pd.factorize(y_ind)
    z_ind, _ = pd.factorize(z_ind)
    d_d, _ = pd.factorize(dc[3])
    return x_ind, y_ind, z_ind, d_d


def load_qct(directory,studies):
    # directory: main directory of qct data
    # studies: list of study names to be included, matching folder names
    for i in studies:
        tib_min = pd.read_csv((directory+i+'_tb_mineral.csv'))
        tib_str = pd.read_csv((directory+i+'_tb_strength.csv'))
        tib_ram = np.concatenate((
            np.array(tib_str.iloc[:,7:]),np.array(tib_min.iloc[:,3:])),axis=1)
        tib_id_ram = np.array(tib_min[['Sub','Trial','Side']])
        
        fem_min = pd.read_csv((directory+i+'_fm_mineral.csv'))
        fem_str = pd.read_csv((directory+i+'_fm_strength.csv'))
        fem_ram = np.concatenate((
            np.array(fem_str.iloc[:,7:]),np.array(fem_min.iloc[:,3:])),axis=1)
        fem_id_ram = np.array(fem_min[['Sub','Trial','Side']])
        
        if 'tib' in locals():
            tib = np.concatenate((tib,tib_ram),axis=0)
            fem = np.concatenate((fem,fem_ram),axis=0)
            tib_id = np.concatenate((tib_id,tib_id_ram),axis=0)
            fem_id = np.concatenate((fem_id,fem_id_ram),axis=0)
        else: 
            tib = tib_ram
            fem = fem_ram
            tib_id = tib_id_ram
            fem_id = fem_id_ram
    return tib, tib_id, fem, fem_id



def load_masks(directory,com_name,ids,mat_shape,res,high_res=4,z_reduction=0.2):
    # com_name: common name for the mask data eg. '_fm_integral.txt'
    # ids: (x,3) matrix of [subject ID, scan #, L/R (L=1, R=2)]
    # mat_shape: shape of the matrix to be loaded (maxes)
    # res: mm per pixel
    # high_res: magnification of high_res data
    # z_reduction: factor to reduce the high res mask to
    c1 = 0
    c2 = 0
    for i in pd.unique(ids[:,0]): # Find all viable data
        for j in range(1,3):
            for k in range(1,4):
                a = directory+str(i)+'_'+str(k)+'_'+str(j)+com_name
                if os.path.exists(a):
                    c1+=1
    # Create mask data shell based on # of samples found
    mask_data = np.full((int(mat_shape[0]//res),
                          int(mat_shape[1]//res),
                          int(mat_shape[2]//res),c1),0)
    higher_res_data = np.full((int(mat_shape[0] // (res / high_res)),
                              int(mat_shape[1] // (res / high_res)),
                              int(mat_shape[2]//res), c1), 0)
    for i in pd.unique(ids[:,0]):
        for j in range(1,3): # Side L / R
            for k in range(1,4): # Scan time point 1, 2, 3
                a = directory+str(i)+'_'+str(k)+'_'+str(j)+com_name
                if os.path.exists(a):
                    # Run registration function to find set of 
                    # x, y, z indices for our matrix, and their corresponding
                    # densities as d_d
                    [x_ind, y_ind, z_ind, d_d] = mask_registration(a,res,c2,1)
                    # Add to the data!
                    np.add.at(mask_data, (x_ind, y_ind, z_ind, c2), d_d)
<<<<<<< Updated upstream
                    [x_ind, y_ind, z_ind, d_d] = mask_registration(a,res/high_res,c2,z_reduction)
=======
                    [x_ind, y_ind, z_ind, d_d] = mask_registration(
                        a,res/high_res,c2,z_reduction)
>>>>>>> Stashed changes
                    np.add.at(higher_res_data, (x_ind, y_ind, z_ind, c2), d_d)
                    c2+=1
                    print('Progress: '+str(c2)+' out of '+str(c1))
    return mask_data, higher_res_data
