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
import pickle
import requests
# Specify the URL of the file you want to download
file_url = 'http://xct.wpi.edu/DISK2/MICROCT/DATA/'
cdvr = '00000009'
meas = '00000073'
file = 'C0000318_1'

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
#%%

download_dicoms('',cdvr,meas,file,file_url)
       #%% 
name = 'calibration'
save = False
plot = True
prnt = False
# fetch the path to the test data
directory = ('C:/Users/arwilzman/OneDrive - Worcester Polytechnic Institute'+
             ' (wpi.edu)/Documents/Desktop/Compare_Registered_DICOMs/'+
             name + '/')
path = (directory + file + '_')
all_items = os.listdir(directory)
file_count = len([_ for _ in os.scandir(directory) if _.is_file()])

for i in range(file_count-1):
    num = str(i).zfill(5)
    ds = dcmread(path+num+'.DCM')
    pat_name = ds.PatientName
    count=0
    if i == 0:
        resolution = float(ds.PixelSpacing[0])
        pointcloud = pd.DataFrame(columns=['x','y','z','d'])
    for x in range(ds.pixel_array.shape[1]):
        for y in range(ds.pixel_array.shape[0]):
            if ds.pixel_array[y,x] < 10:
                continue
            pc_ram = pd.DataFrame([[x*resolution,y*resolution,
                                   float(ds.SliceLocation),ds.pixel_array[y,x]]],
                                  columns=['x','y','z','d'])
            pointcloud = pd.concat((pointcloud,pc_ram))
        
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
if save:
    with open(name+'.pkl','wb') as file:
        pickle.dump(pointcloud,file)
    pointcloud.to_csv(name+'.csv')
    