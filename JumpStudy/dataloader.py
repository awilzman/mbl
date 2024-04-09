# -*- coding: utf-8 -*-
#%% Initalize
"""
Created on Thu Sep 14 10:24:11 2023

@author: arwilzman
Requires the following data: 
    
    
    
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.signal import butter, filtfilt
from tabulate import tabulate

date = datetime.now()
month = date.strftime("%m")
day = date.strftime("%d")
year = date.strftime("%y")
date = ('_'+month + '_' + day + '_' + year)

plot_fig = True
lots_plots = True
save_fig = True
reload_data = True
save_data = True

first_load = False #ONLY REQUIRED IF STATIC OPTIMIZATION WAS RERUN FOR ANY TRIAL

directory = 'Z:/_Current IRB Approved Studies/Jumping_Study/'
study = 'JS'
prefix = ['S','F'] #For joint angle search
trialtypes = ['SLJump','DropJump']
max_pts = 20
max_trials = 25

code = range(1,max_pts+1)
tr_names=trialtypes
trials = range(1,max_trials+1)
# =============================================================================

def read_mot(file,directory):
    # only read in kinematics during force plate activation!!!
    file_path = directory + file
    df = None
    with open(file_path, "r") as mot_file:
        lines = mot_file.readlines()
    header_lines = 0
    for line in lines:
        if "endheader" in line.lower():
            header_lines += 1
            break
        else:
            header_lines += 1
    data_lines = lines[header_lines:]
    header = ['time','pelvis_tx','pelvis_ty','pelvis_tz','pelvis_tilt',
              'pelvis_list','pelvis_rotation','hip_flexion_r',
              'hip_adduction_r','hip_rotation_r','knee_angle_r',
              'ankle_angle_r','subtalar_angle_r','mtp_angle_r',
              'hip_flexion_l','hip_adduction_l','hip_rotation_l',
              'knee_angle_l','ankle_angle_l','subtalar_angle_l',
              'mtp_angle_l','lumbar_extension','lumbar_bending',
              'lumbar_rotation','r_ground_force_vx','r_ground_force_vy',
              'r_ground_force_vz','r_ground_force_px','r_ground_force_py',
              'r_ground_force_pz','ground_force_vx','ground_force_vy',
              'ground_force_vz','ground_force_px','ground_force_py',
              'ground_force_pz','ground_torque_x','ground_torque_y',
              'ground_torque_z','ground_torque_x','ground_torque_y','ground_torque_z']
    data = [i.strip('\n').split('\t') for i in data_lines[1:]]
    data_float = [[float(value) for value in row] for row in data]
    df = pd.DataFrame(data_float, columns=header)
    start_index = df[df['r_ground_force_vy']==df['r_ground_force_vy'].max()].index
    start_index -= 30
    end_index = start_index + 100
    start_index = max(0, start_index.min()) #don't get lower than 0
    end_index = min(df.index.max(), end_index.max()) #don't go higher than the end
    condition_to_drop = ((df.index <= start_index) | (df.index >= end_index))
    df = df.loc[~condition_to_drop]
    df = df.reset_index()
    return df
        
def reform_sto(file_path,new_file_path):
    with open(file_path, "r") as sto_file:
        lines = sto_file.readlines()  # Read all lines into a list
    # Identify the end of the header section using "endheader" tag
    header_lines = 0
    for line in lines:
        if "endheader" in line.lower():  # Case-insensitive check
            header_lines += 1
            break  
        else:
            header_lines += 1
    data_lines = lines[header_lines:]
    with open(new_file_path, "w") as new_sto_file:
        new_sto_file.writelines(data_lines)
    os.remove(file_path)

def bulk_renamer(directory = ('Z:/_Current IRB Approved Studies/Jumping_Study/'),
                 study = 'JS',prefix = ['S','F'],
                 code = range(1,21),BL_name = 'DropJump',UL_name = 'SLJump',
                 trial = range(1,21),suffix = '_JointReaction_ReactionLoads.sto'):
    for c in code:
        for p in prefix:
            a=study+'_'+p+str(c)
            first_check = (directory+a+'/OpenSim_Results/'+a+'_'+UL_name+'3'+suffix)
            if os.path.exists(first_check):
                for t in trial:
                    BLt = (directory+a+'/OpenSim_Results/'+a+'_'+BL_name+str(t)+suffix)
                    ULt = (directory+a+'/OpenSim_Results/'+a+'_'+UL_name+str(t)+suffix)
                    if os.path.exists(BLt):
                        new_BL_name = BLt.split('/')[-1]
                        new_BL_dir = '/'.join(BLt.split('/')[:-1]) + '/'
                        new_BL_name = new_BL_name.replace('_JointReaction_ReactionLoads','')
                        reform_sto(BLt,new_BL_dir+new_BL_name)
                    if os.path.exists(ULt):
                        new_UL_name = ULt.split('/')[-1]
                        new_UL_dir = '/'.join(ULt.split('/')[:-1]) + '/'
                        new_UL_name = new_UL_name.replace('_JointReaction_ReactionLoads','')
                        reform_sto(ULt,new_UL_dir+new_UL_name)

def read_sto(file_path):
    with open(file_path, "r") as sto_file:
        # Read the data lines into a Pandas DataFrame, starting from the second line
        df = pd.read_csv(file_path, delimiter='\t', header=None,skiprows=[0])
        return df

def bulk_loader(directory,study,prefix,code,tr_names,trials,suffix='.sto'):
    # Used to pull data from OpenSim ReactionLoads.sto files that have been
    # sifted through by bulk_renamer. Designed for two-condition comparisons
    # Requirements:
    #   participant naming scheme: {study}_{prefix}{code}
    #   file structure
    The_Data = pd.DataFrame()
    header = ['time','delete','delete','delete','delete','delete','delete','delete',
              'delete','delete','Right Hip Force X','Right Hip Force Y','Right Hip Force Z',
              'Right Hip Moment X','Right Hip Moment Y','Right Hip Moment Z',
              'Right Hip pX ref Pelvis','Right Hip pY ref Pelvis','Right Hip pZ ref Pelvis',
              'Right Knee Force X','Right Knee Force Y','Right Knee Force Z',
              'Right Knee Moment X','Right Knee Moment Y','Right Knee Moment Z',
              'Right Knee pX ref Femur','Right Knee pY ref Femur','Right Knee pZ ref Femur',
              'Right Ankle Force X','Right Ankle Force Y','Right Ankle Force Z',
              'Right Ankle Moment X','Right Ankle Moment Y','Right Ankle Moment Z',
              'Right Ankle pX ref Tibia','Right Ankle pY ref Tibia','Right Ankle pZ ref Tibia',
              'Right Subtalar Force X','Right Subtalar Force Y','Right Subtalar Force Z',
              'Right Subtalar Moment X','Right Subtalar Moment Y','Right Subtalar Moment Z',
              'Right Subtalar pX ref Talus','Right Subtalar pY ref Talus','Right Subtalar pZ ref Talus',
              'delete','delete','delete','delete','delete','delete','delete','delete','delete',
              'Left Hip Force X','Left Hip Force Y','Left Hip Force Z',
              'Left Hip Moment X','Left Hip Moment Y','Left Hip Moment Z',
              'Left Hip pX ref Pelvis','Left Hip pY ref Pelvis','Left Hip pZ ref Pelvis',
              'Left Knee Force X','Left Knee Force Y','Left Knee Force Z',
              'Left Knee Moment X','Left Knee Moment Y','Left Knee Moment Z',
              'Left Knee pX ref Femur','Left Knee pY ref Femur','Left Knee pZ ref Femur',
              'Left Ankle Force X','Left Ankle Force Y','Left Ankle Force Z',
              'Left Ankle Moment X','Left Ankle Moment Y','Left Ankle Moment Z',
              'Left Ankle pX ref Tibia','Left Ankle pY ref Tibia','Left Ankle pZ ref Tibia',
              'Left Subtalar Force X','Left Subtalar Force Y','Left Subtalar Force Z',
              'Left Subtalar Moment X','Left Subtalar Moment Y','Left Subtalar Moment Z',
              'Left Subtalar pX ref Talus','Left Subtalar pY ref Talus','Left Subtalar pZ ref Talus',
              'delete','delete','delete','delete','delete','delete','delete',
              'delete','delete','delete','delete','delete','delete','delete',
              'delete','delete','delete','delete','RVL IMU','RVM IMU','RBF IMU',
              'RST IMU','RGlM IMU','RGL IMU','RSL IMU','RTA IMU','LVL IMU',
              'LVM IMU','LBF IMU','LST IMU','LGlM IMU','LGL IMU','LSL IMU','LTA IMU',
              'ID','Sex','Mass','Trial No','Tibia Stiffness',
              'Height','Landing Limbs']
    imus = []
    for c in code:
        for p in prefix:
            a = study + '_' + p + str(c)
            height_file = os.path.join(directory+a+'/heights.txt')
            if os.path.exists(height_file):
                heights = pd.read_csv(height_file,delimiter='\t').drop(index = range(4))
                stiff_file = f'{directory}{a}/FE/FE_STD.TXT'
                if os.path.exists(os.path.join(stiff_file)):
                    stiffness = pd.read_csv(stiff_file,delimiter='\t')['S']
                    stiffness = stiffness[0] 
                else:
                    stiffness = None
                imus = pd.read_excel(os.path.join(directory,'Jump Study Participants.xlsx'),sheet_name='EMG Locations')
                imus = imus[imus['Participant']==a]
                dem = pd.read_excel(os.path.join(directory,'Jump Study Participants.xlsx'),sheet_name='Overview')
                sex = dem[dem['ID']==c]['Sex'].iloc[0]
                mass = dem[dem['ID']==c]['Mass (kg)'].iloc[0]
                for t in trials:
                    for data_file in tr_names:
                        jcf_file = (directory+a+'/'+'OpenSim_Results/JCF/'+a+'_'+
                                    f'{data_file}{str(t)}{suffix}')
                        if os.path.exists(jcf_file):
                            ram = read_sto(jcf_file)
                            ram['RVL IMU'] = imus.iloc[0]['R VL']
                            ram['RVM IMU'] = imus.iloc[0]['R VM']
                            ram['RBF IMU'] = imus.iloc[0]['R BF']
                            ram['RST IMU'] = imus.iloc[0]['R ST']
                            ram['RGlM IMU'] = imus.iloc[0]['R GlM']
                            ram['RGL IMU'] = imus.iloc[0]['R GL']
                            ram['RSL IMU'] = imus.iloc[0]['R SL']
                            ram['RTA IMU'] = imus.iloc[0]['R TA']
                            ram['LVL IMU'] = imus.iloc[0]['L VL']
                            ram['LVM IMU'] = imus.iloc[0]['L VM']
                            ram['LBF IMU'] = imus.iloc[0]['L BF']
                            ram['LST IMU'] = imus.iloc[0]['L ST']
                            ram['LGlM IMU'] = imus.iloc[0]['L GlM']
                            ram['LGL IMU'] = imus.iloc[0]['L GL']
                            ram['LSL IMU'] = imus.iloc[0]['L SL']
                            ram['LTA IMU'] = imus.iloc[0]['L TA']
                            ram['ID'] = c
                            ram['Sex'] = sex
                            ram['Mass'] = mass
                            ram['Trial No'] = t
                            ram['Tibia Stiffness'] = stiffness
                            ram['Height'] = heights[(data_file+str(t)+'.c3d')].dropna().to_numpy().astype(float).mean()
                            if data_file == 'SLJump':
                                ram['Landing Limbs'] = 1
                            else:
                                ram['Landing Limbs'] = 2
                            The_Data = pd.concat([The_Data,ram], ignore_index=True)
    The_Data.columns = header
    mask = The_Data.columns.str.startswith('delete')
    The_Data = The_Data.drop(columns=The_Data.columns[mask])
    return The_Data

# Function to apply a low-pass filter to the data
def apply_low_pass_filter(time,data_tf,cutoff=10,sampling_rate=None):
    if sampling_rate is None:
        loc1 = 1
        loc2 = 0
        while time.iloc[loc1]-time.iloc[loc2] <= 0:
            loc1 += 1
            loc2 += 1
        sampling_rate = 1 / (time.iloc[1] - time.iloc[0])  # Calculate sample rate
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False) # Butter it up
    filtered_data = filtfilt(b, a, data_tf)
    return filtered_data

def max_loading_rate(time,force):
    lr = force.diff() / time.diff()
    return lr.min() #loading direction is flipped for us, negative = compressive

def fft_int(time, force, stiff=None, threshold=0.9):
    # Perform FFT on force data and integrate
    
    fft_force = np.fft.fft(force / stiff) if stiff is not None else np.fft.fft(force)
    fft_force = np.abs(fft_force[:len(fft_force)//2 - 1])
    
    fft_freq = np.fft.fftfreq(len(force), d=(time.iloc[1] - time.iloc[0]))
    fft_freq = fft_freq[:len(fft_freq)//2 - 1]
    
    integral = 0
    step = (fft_freq[1] - fft_freq[0]) / 2
    for i, j in enumerate(fft_force):
        integral += j * step
        
    return integral

def di_bin(max_accel,
           start_level=1.3,end_level=10.8,low_g_interval=0.2,max_interval=0.6):
    # Calculate the number of bins
    num_bins = int((end_level - start_level) / low_g_interval) + 1
    # Generate the list of bins
    bins = np.arange(start_level, end_level, low_g_interval)
    greater_than = [i for i in bins if i < max_accel]
    return len(greater_than)

def daily_impact(directory,pt,tri,sensor_number):
    files = os.listdir(os.path.join(directory,pt,'EMG'))
    file = [f for f in files if tri in f]
    emg_file = os.path.join(directory, pt, 'EMG', file[0])
    targets = [
        f'Avanti sensor {sensor_number}: Acc {sensor_number}.X (IM)',
        f'Avanti sensor {sensor_number}: Acc {sensor_number}.Y (IM)',
        f'Avanti sensor {sensor_number}: Acc {sensor_number}.Z (IM)',
        f'Trigno sensor {sensor_number}: Acc {sensor_number}.X (IM)',
        f'Trigno sensor {sensor_number}: Acc {sensor_number}.Y (IM)',
        f'Trigno sensor {sensor_number}: Acc {sensor_number}.Z (IM)',
        f'Trigno sensor {sensor_number}: Acc {sensor_number}.X',
        f'Trigno sensor {sensor_number}: Acc {sensor_number}.Y',
        f'Trigno sensor {sensor_number}: Acc {sensor_number}.Z']
    
    selected_data_list = []
    flag = True
    with open(emg_file, "r") as efile:
        lines = efile.readlines()  
    header_lines = 0
    for line in lines:
        if "x[s]" in line.lower():  # Case-insensitive check
            break  
        else:
            header_lines += 1
    emg = pd.read_csv(emg_file,skiprows=header_lines)
    for t in emg.columns:
        if any(sensor_type in t for sensor_type in ['Avanti', 'Trigno']):
            # Select the target column and the column to its left (time)
            if flag:
                selected_columns = [emg.columns[emg.columns.get_loc(t) - 1], t]
                flag = False
            else:
                selected_columns = t
            # Create a new DataFrame with the selected columns
            selected_data = emg[selected_columns]
            selected_data_list.append(selected_data)
    # Combine the selected DataFrames into a single DataFrame
    data = pd.concat(selected_data_list, axis=1)
    if targets[0] in data.columns:
        accel_magnitude = np.sqrt(data[targets[0]]**2 + data[targets[1]]**2 + data[targets[2]]**2)
    elif targets[3] in data.columns:
        accel_magnitude = np.sqrt(data[targets[3]]**2 + data[targets[4]]**2 + data[targets[5]]**2)
    elif targets[6] in data.columns:
        accel_magnitude = np.sqrt(data[targets[6]]**2 + data[targets[7]]**2 + data[targets[8]]**2)
    else:
        return None, None
    data = pd.concat([data, pd.DataFrame({'Accel': accel_magnitude})], axis=1)
    cond = (data['Accel']>0)
    accel = apply_low_pass_filter(data[cond][data.columns[0]],data[cond]['Accel'],cutoff=20,sampling_rate=1000)
    max_accel = accel.max()
    fft_accel = fft_int(data[cond][data.columns[0]],accel)    
    return di_bin(max_accel),fft_accel

#%% Load Data
if first_load:
    print('Renaming any ReactionLoads files left by OpenSim')
    bulk_renamer()
if reload_data:
    print('Loading data')
    The_Data = bulk_loader(directory,study,prefix,code,tr_names,trials)
    The_Data = The_Data.reset_index(drop=True)
    The_Data['MassN'] = The_Data['Mass'] * 9.81
    The_Data.to_pickle(study+date+'.pkl')
else:
    print('Reading initial data from pickle!')
    The_Data = pd.DataFrame()
    The_Data = pd.read_pickle(study+date+'.pkl')
    
The_List=The_Data[['ID', 'Mass','Height','Landing Limbs','Trial No']].drop_duplicates()
The_List=The_List.reset_index(drop=True)
The_Data=The_Data.reset_index(drop=True)

columns_to_filter = ['Right Ankle Force Y','Right Knee Force Y']
print('Processing data...')
for index, trial in The_List.iterrows():
    condition = ((The_Data['ID'] == trial['ID']) & 
                 (The_Data['Mass'] == trial['Mass']) & 
                 (The_Data['Height'] == trial['Height']) & 
                 (The_Data['Landing Limbs'] == trial['Landing Limbs']) & 
                 (The_Data['Trial No'] == trial['Trial No']))
    for c in columns_to_filter:
        ram = apply_low_pass_filter(The_Data['time'][condition],The_Data[c][condition])
        The_Data.loc[condition, c] = ram
    # Find the index of peak 'Right Ankle Force Y'
    ram = The_Data[condition]['Right Ankle Force Y'].min()
    start_index = The_Data[condition & (The_Data['Right Ankle Force Y'] == ram)].index
    start_index -= 30 # get 0.3 s before
    end_index = start_index + 100 #total 1.0 s trial time
    start_index = max(0, start_index.min()) #don't get lower than 0
    end_index = min(The_Data.index.max(), end_index.max()) #don't go higher than the end
    condition_to_drop = ((The_Data.index <= start_index) | (The_Data.index >= end_index)) & condition
    The_Data = The_Data.loc[~condition_to_drop]

# Create height groups
The_List['Height groups'] = np.floor(The_List['Height']*10)
The_List.loc[The_List['Height groups']<4,'Height groups'] = 2
# Reassign height -- original measure is based on marker location, but this
# code re-asserts it to what I know happened during the data collection.
The_List['Height'] = The_List['Height groups'] * 0.1
The_List.drop('Height groups',axis=1,inplace=True)
#%% Visualize Data
interest = ['Right Ankle Force Y','Right Knee Force Y']
conditions = [(The_Data['Landing Limbs']==2),(The_Data['Landing Limbs']==1)]
for i in interest:
    fig, ax = plt.subplots(figsize=(8, 6))
    X = The_Data['Mass']
    Y = The_Data[i] / (The_Data['MassN'])
    limbs = The_Data['Landing Limbs']
    ax.scatter(X,Y,c=limbs, marker='x', cmap='cool', alpha=0.05)
    ax.set_title(i,fontsize=20)
    ax.set_ylabel('Contact Force [BW]')
    ax.set_xlabel('Participant Mass [kg]')
    ax.set_ylim([0,-20])
    plt.text(78,-18,'Unilateral', ha='center', va='bottom', fontsize=20, color='cyan')
    plt.text(78,-16.5,'Bilateral', ha='center', va='bottom', fontsize=20, color='fuchsia')
    if save_fig:
        plt.savefig(f'{directory}Data/graphpics/All {i}.png',bbox_inches='tight')
    if plot_fig:
        plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    count=0
    colors = ['fuchsia','aqua']
    for cond in conditions:
        hist, xed, yed = np.histogram2d((-The_Data[cond][i]/The_Data[cond]['MassN']),
                                        The_Data[cond]['Height'],bins=50,range=[[2,25],[0,.7]])
        xpos, ypos = np.meshgrid(xed[:-1]+0.1, yed[:-1]+0.1, indexing='ij')
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        dy = 0.1*np.ones_like(zpos)
        dx = np.ones_like(zpos)
        dz = hist.ravel()
        colr = colors[count]
        count += 1
        ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color=colr,alpha=0.3-count*0.09,antialiased=True)
    ax.set_xlabel(f'{i} (BW)')
    ax.set_ylabel('Drop Height (m)')
    ax.set_title(f'Frequency of {i} > 2 BW')
    ax.set_zlim([0,250])
    ax.text(20,1.3,250,'Unilateral', ha='center', va='bottom', fontsize=18, color='cyan')
    ax.text(20,1.3,200,'Bilateral', ha='center', va='bottom', fontsize=18, color='fuchsia')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{directory}Data/graphpics/Histogram {i}.png',bbox_inches='tight')
    if plot_fig:
        plt.show()
#%% Calculate Metrics and Save
for index, listed_trial in The_List.iterrows():
    kinematics = None
    trialtype = trialtypes[int(listed_trial['Landing Limbs']-1)]
    pt = int(listed_trial['ID'])
    t = int(listed_trial['Trial No'])
    limbs = int(listed_trial['Landing Limbs'])
    
    # figure out prefix and read kinematic data
    for i in prefix:
        file = f'{study}_{i}{pt}/OpenSim_Data/Input/'
        file += f'{study}_{i}{pt}_{trialtype}{t}_input.mot'
        file_path = directory + file
        if os.path.exists(file_path):
            kinematics = read_mot(file, directory)
            person = f'{study}_{i}{pt}'
            break
    condition_data = (
        (The_Data['ID'] == pt) & 
        (The_Data['Landing Limbs'] == limbs) & 
        (The_Data['Trial No'] == t)
    )
    trial_data = The_Data[condition_data]
    mass = trial_data.iloc[0]['MassN']
    tibstiff = trial_data.iloc[0]['Tibia Stiffness']
    # Calculate metrics
    hip_flex_rom = kinematics['hip_flexion_r'].max() - kinematics['hip_flexion_r'].min()
    hip_flex_init = kinematics.iloc[0]['hip_flexion_r']
    knee_flex_rom = kinematics['knee_angle_r'].max() - kinematics['knee_angle_r'].min()
    knee_flex_init = kinematics.iloc[0]['knee_angle_r']
    ankle_flex_rom = kinematics['ankle_angle_r'].max() - kinematics['ankle_angle_r'].min()
    ankle_flex_init = kinematics.iloc[0]['ankle_angle_r']
        
    force = -kinematics['r_ground_force_vy'] / mass
    contact_force = trial_data['Right Ankle Force Y'] / mass
    time = trial_data['time']
    rxf_time = kinematics['time']
    
    max_jcf = contact_force.min()
    max_jcf_lr = max_loading_rate(time, contact_force)
    max_rxf = force.min()
    max_rxf_lr = max_loading_rate(rxf_time, force)
    
    fft_jcf = fft_int(time, contact_force)
    fft_rxf = fft_int(rxf_time, force)
    
# =============================================================================
#     if fft_rxf is not None:
#         if fft_rxf > 2000:
#             fft_rxf = None
#             fft_rxf_fe = None
#             print(f'fft_rxf outlier: {pt}, {trialtype}{t}')
# =============================================================================
    if tibstiff is not None:
        max_jcf_strain = max_jcf * mass / tibstiff
        max_rxf_strain = max_rxf * mass / tibstiff
        fft_jcf_fe = fft_int(time, contact_force, tibstiff)
        fft_rxf_fe = fft_int(rxf_time, force, tibstiff)
    # force is in bodyweights currently., needed to switch back
    # get strain rates
        max_jcf_strmagrate = abs(max_jcf_strain * max_jcf_lr)
        max_rxf_strmagrate = abs(max_rxf_strain * max_rxf_lr)
        
    if trial_data.iloc[0]['RTA IMU'] == 0:
        d_imp = None
        imu_fft = None
    else:
        d_imp, imu_fft = daily_impact(directory, person, f'{trialtype}{t}', trial_data.iloc[0]['RTA IMU'])

    # Assign metrics
    condition_list = (
        (The_List['ID'] == pt) & 
        (The_List['Landing Limbs'] == limbs) & 
        (The_List['Trial No'] == t))
    The_List.loc[condition_list, 'Hip Flexion ROM'] = hip_flex_rom
    The_List.loc[condition_list, 'Knee Flexion ROM'] = knee_flex_rom
    The_List.loc[condition_list, 'Ankle Flexion ROM'] = ankle_flex_rom
    The_List.loc[condition_list, 'Hip Flexion at Contact'] = hip_flex_init
    The_List.loc[condition_list, 'Knee Flexion at Contact'] = knee_flex_init
    The_List.loc[condition_list, 'Ankle Flexion at Contact'] = ankle_flex_init
    The_List.loc[condition_list, 'RXF'] = abs(max_rxf)
    The_List.loc[condition_list, 'RXF_R'] = abs(max_rxf_lr)
    The_List.loc[condition_list, 'JCF'] = abs(max_jcf)
    The_List.loc[condition_list, 'JCF_R'] = abs(max_jcf_lr)
    The_List.loc[condition_list, 'RXF_FE_FFT'] = abs(fft_rxf_fe)
    The_List.loc[condition_list, 'JCF_FE_FFT'] = abs(fft_jcf_fe)
    The_List.loc[condition_list, 'RXF_FFT'] = abs(fft_rxf)
    The_List.loc[condition_list, 'JCF_FFT'] = abs(fft_jcf)
    The_List.loc[condition_list, 'RXF_FE'] = abs(max_rxf_strain)
    The_List.loc[condition_list, 'JCF_FE'] = abs(max_jcf_strain)
    The_List.loc[condition_list, 'RXF_SMR'] = abs(max_rxf_strmagrate)
    The_List.loc[condition_list, 'JCF_SMR'] = abs(max_jcf_strmagrate)
    The_List.loc[condition_list, 'DIS'] = abs(d_imp) if d_imp is not None else None
    The_List.loc[condition_list, 'IMU_FFT'] = abs(imu_fft) if imu_fft is not None else None
    #https://doi.org/10.1002/jbmr.3999
    #https://doi.org/10.1016/j.jbiomech.2010.03.021
    
    if lots_plots:
        if save_fig | plot_fig:
            plt.scatter(trial_data['time'],
                        trial_data['Right Ankle Force Y']/trial_data['MassN'])
            #plt.title(f'{person} {trialtype} {t}', fontsize=20)
            plt.ylabel(f'Contact Force (BW)', fontsize=20)
            plt.xlabel('time (s)', fontsize=20)
            plt.ylim([0,-20])
            plt.tight_layout()
        if save_fig:
            plt.savefig(f'{directory}/Data/graphpics/{person}_{trialtype}_{t}_RANK_JCF.png',bbox_inches='tight')
        if plot_fig:
            plt.show()
summary = The_List.describe()
table_float_format = '.4e'
table = tabulate(summary,headers='keys',floatfmt=table_float_format,tablefmt='pretty')

with open(f'{directory}Data/Jump_Study_DataSummary{date}.txt', "w") as file:
    file.write(table)
if save_data:
    The_List.to_csv(directory+f'Data/Jump_Study_Data{date}.csv',index=False, float_format='%.9e')
    The_List.to_excel(directory+f'Data/Jump_Study_Data{date}.xlsx',index=False,
                      sheet_name='Sheet1', engine='xlsxwriter', float_format='%.9e')