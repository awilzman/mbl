# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:23:05 2023

@author: arwilzman
"""

#%% Import
from pydicom import dcmread
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.decomposition import IncrementalPCA
from scipy.spatial.distance import cdist
# Specify the URL of the file you want to download
file_url = 'http://xct.wpi.edu/DISK2/MICROCT/DATA/'
cdvr = '00000561'
meas = '00004141'
file = 'C0004382'
name = 'test_sub'
threshold = 5000
trunc = 100
save = True
plot = True
prnt = True
calibrate_slope = 0.000357
calibrate_int = -0.0012625
# fetch the path to the test data
local_directory = 'C:/Users/arwilzman/OneDrive - Worcester Polytechnic Institute (wpi.edu)'
local_directory += '/Documents/Desktop/Compare_Registered_DICOMs/'
directory = (local_directory+name+'/')
path = (directory+file)

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
if not os.path.exists(directory):
    os.makedirs(directory)
    print('hold up, need to grab these for you')
    download_dicoms(directory,cdvr,meas,file,file_url)
    
if os.path.exists(name+'.pkl'):
    output = pd.read_pickle(name+'.pkl')
    save = False
    print('Loaded from pickle')
else:
    all_items = os.listdir(directory)
    file_count = len([_ for _ in os.scandir(directory) if _.is_file()])
    for i in range(file_count - 1):
        num = str(i).zfill(5)
        ds = dcmread(f'{path}_{num}.DCM')
        pat_name = ds.PatientName
        if i == 0:
            resolution = float(ds.PixelSpacing[0])
            output = pd.DataFrame(columns=['x','y','z','d'])
        # Apply threshold
        indices = np.where(ds.pixel_array >= threshold)
        z_values = np.array([ds.SliceLocation] * len(indices[0]))
        df = pd.DataFrame({'x': indices[1][::trunc] * resolution,
                           'y': indices[0][::trunc] * resolution,
                           'z': z_values[::trunc],
                           'd': ds.pixel_array[indices][::trunc]})
        # Update the 'output' DataFrame directly
        df['d'] = df['d']*calibrate_slope+calibrate_int
        output = pd.concat([output,df], axis=0)
        if prnt:
            print(f"Patient's Name...: {pat_name.family_comma_given()}")
            print(f"Patient ID.......: {ds.PatientID}")
            print(f"Modality.........: {ds.Modality}")
            print(f"Study Date.......: {ds.StudyDate}")
            print(f"Image size.......: {ds.Rows} x {ds.Columns}")
            print(f"Pixel Spacing....: {ds.PixelSpacing}")
            print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")
        print(f"Progress...: {i+1} / {file_count}")
        if plot:
            plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
            plt.show()
    output.reset_index(drop=True, inplace=True)
if save:
    output.to_pickle(name + '.pkl')
    output[['x','y','z']].round(3).to_csv(
        name+'.txt',index=False,header=False,sep='\t',float_format='%.3f')
    