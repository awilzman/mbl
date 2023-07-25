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
from arw_training import mask_registration
def load_qct(directory,studies):
    # directory: main directory of qct data
    # studies: list of study names to be included, matching folder names
    for i in studies:
        tib_min = pd.read_csv((directory+i+'_tb_mineral.csv'))
        tib_str = pd.read_csv((directory+i+'_tb_strength.csv'))
        tib_ram = np.concatenate((np.array(tib_str.iloc[:,7:]),np.array(tib_min.iloc[:,3:])),axis=1)
        tib_id_ram = np.array(tib_min[['Sub','Trial','Side']])
        
        fem_min = pd.read_csv((directory+i+'_fm_mineral.csv'))
        fem_str = pd.read_csv((directory+i+'_fm_strength.csv'))
        fem_ram = np.concatenate((np.array(fem_str.iloc[:,7:]),np.array(fem_min.iloc[:,3:])),axis=1)
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
                              int(mat_shape[2] * z_reduction // (res / high_res))+10, c1), 0)
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
                    np.add.at(higher_res_data, (x_ind, y_ind, z_ind, c2), d_d)
                    c2+=1
                    print('Progress: '+str(c2)+' out of '+str(c1))
    return mask_data, higher_res_data
