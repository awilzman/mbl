#%% Import
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn import preprocessing
import numpy as np
import torch
import time
from torch import nn
import os
if torch.cuda.is_available():
    print('CUDA available')
    print(torch.cuda.get_device_name(0))
else:
    print('CUDA *not* available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from arw_training import *
from SCI_dataloader import *

import datetime

date = datetime.datetime.now()
month = date.strftime("%m")
day = date.strftime("%d")
year = date.strftime("%y")
date = ('_'+month + '_' + day + '_' + year)

#%% Load QCT data
# 
directory = 'Z:/_Lab Personnel Folders/Andrew/Codes/SCI/'
studies = ['Feb28_SCI_rowing','Ekso','Healthy']

[tib, tib_id, fem, fem_id] = load_qct(directory,studies)

tib[np.isnan(tib)] = 0
fem[np.isnan(fem)] = 0

# scale each row
scaler = preprocessing.MinMaxScaler(feature_range=(0,10))
transformer = np.concatenate((tib,fem),axis=0)
t_minmax = np.zeros((2,65)) #only 65 variables

# store scaler data in t_minmax
for i in range(65):
    transformer[:,i] = scaler.fit_transform(transformer[:,i].reshape(-1,1))[:,0]
    t_minmax[0,i] = scaler.data_min_
    t_minmax[1,i] = scaler.data_max_
    
# separate tibia and femur data
tib = transformer[0:len(tib),:]
fem = transformer[len(tib):,:]

# initialize key variables
seed = 1
sz = np.size(fem,1) 
maxes = [200,200,300] # [x,y,z] mm, will throw errors if this isn't big enough
target_resolution = 4.0 # mm per element
high_res_mag = 3
high_res_z_reduction = 0.3
#%% Femur mask load
#bme80 
directory = 'C:/Users/arwilzman/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/masks/'
#bme81 directory = 'C:/Users/arwilzman/Desktop/DL_Data/Mask_data_move_to_local/'
fem_mask_data, HR_fem_mask_data = load_masks(directory,'_fm_integral.txt',fem_id,
                                             maxes,target_resolution,high_res_mag,
                                             high_res_z_reduction)

                        
#%% Tibia mask load
#bme80 
directory = 'C:/Users/arwilzman/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/masks/tibia/'
#bme81 directory = 'C:/Users/arwilzman/Desktop/DL_Data/Mask_data_move_to_local/'
tib_mask_data, HR_tib_mask_data = load_masks(directory,'_tb_integral.txt',tib_id,
                                             maxes,target_resolution,high_res_mag,
                                             high_res_z_reduction)

#%% Initialize Networks

# initalize required parameters 
trunc = 3 # number of parameters to reduce each region to in the informed QCT net
layer1_size = 40 # uninformed network first layer size
layer2_size = 8 # uninformed network second layer size
# the following are indices of QCT metrics for each bone region
epi_ind = [[0,3,6,11,14,15,16,17,18,19,20,21,22,50,51,52,62]]
met_ind = [[1,4,7,12,23,24,25,26,27,28,29,30,31,53,54,55,63]]
dia_ind = [[2,5,8,13,32,33,34,35,36,37,38,39,40,56,57,58,64]]
tot_ind = [[9,10,41,42,43,44,45,46,47,48,49,59,60,61]]

#import networks
from bone_networks import *
#%% load and train informed networks
num_epochs = 1000
learn_rate = 1e-4
weight_decay = 1e-5
batch_size = 70
seed=100
p_int = num_epochs//10
torch.manual_seed(seed)
data = transformer
# report last model saved date with leading _
last_date = '_07_14_23'


all_nn = informed_net(epi_ind, met_ind,dia_ind,tot_ind,trunc,layer2_size).cuda()
itib_nn = informed_net(epi_ind, met_ind,dia_ind,tot_ind,trunc,layer2_size).cuda()
ifem_nn = informed_net(epi_ind, met_ind,dia_ind,tot_ind,trunc,layer2_size).cuda()
crit = nn.MSELoss()
all_nn.load_state_dict(torch.load('iall'+last_date+'.pt'))
itib_nn.load_state_dict(torch.load('itib'+last_date+'.pt'))
ifem_nn.load_state_dict(torch.load('ifem'+last_date+'.pt'))

# retrain and report networks
iall_net, all_losses = train_unsupervised(data, all_nn, num_epochs, learn_rate, 
                                          weight_decay, batch_size*2, crit, p_int, device)
[MAE_all,MSE_all,RMSE_all,R2_all]=model_eval_unsupervised(data,iall_net)
code_all = 'i_all, MSE, parameterized  ' + str(trunc*4) + ',' + str(layer2_size)
itib_net, itib_losses = train_unsupervised(tib, itib_nn, num_epochs, learn_rate, 
                                           weight_decay, batch_size, crit, p_int, device)
[MAE_itib,MSE_itib,RMSE_itib,R2_itib]=model_eval_unsupervised(tib,itib_net)
code_itib = 'i_tib, MSE, parameterized  ' + str(trunc*4) + ',' + str(layer2_size)
ifem_net, ifem_losses = train_unsupervised(fem, ifem_nn, num_epochs, learn_rate,
                                           weight_decay, batch_size, crit, p_int, device)
[MAE_ifem,MSE_ifem,RMSE_ifem,R2_ifem]=model_eval_unsupervised(fem,ifem_net)
code_ifem = 'i_fem, MSE, parameterized  ' + str(trunc*4) + ',' + str(layer2_size)

all_pred = iall_net.decode(iall_net.encode(torch.FloatTensor(data).cuda())).cpu().detach().numpy()
tib_pred = itib_net.decode(itib_net.encode(torch.FloatTensor(tib).cuda())).cpu().detach().numpy()
fem_pred = ifem_net.decode(ifem_net.encode(torch.FloatTensor(fem).cuda())).cpu().detach().numpy()
plot_network(code_all,iall_net,data,all_pred,all_losses,MAE_all,RMSE_all,R2_all)
plot_network(code_itib,itib_net,tib,tib_pred,itib_losses,MAE_itib,RMSE_itib,R2_itib)
plot_network(code_ifem,ifem_net,fem,fem_pred,ifem_losses,MAE_ifem,RMSE_ifem,R2_ifem)

#save networks
torch.save(iall_net.state_dict(),'iall'+date+'.pt')
torch.save(ifem_net.state_dict(),'ifem'+date+'.pt')
torch.save(itib_net.state_dict(),'itib'+date+'.pt')

#%% load fully connected networks

last_date = '_07_24_23'


all_nn = FC_net(layer1_size,layer2_size,sz).cuda()
tib_nn = FC_net(layer1_size,layer2_size,sz).cuda()
fem_nn = FC_net(layer1_size,layer2_size,sz).cuda()
crit = nn.MSELoss()
all_nn.load_state_dict(torch.load('all_net'+last_date+'.pt'))
tib_nn.load_state_dict(torch.load('tib_net'+last_date+'.pt'))
fem_nn.load_state_dict(torch.load('fem_net'+last_date+'.pt'))

#%% train fully connected networks

num_epochs = 100
p_int = num_epochs//10
learn_rate = 1e-4
weight_decay = 1e-5
batch_size = 70
seed=1237769
torch.manual_seed(seed)
data = transformer

all_net, all_losses = train_unsupervised(data, all_nn, num_epochs, learn_rate, 
                                         weight_decay, batch_size*2, crit, p_int, device)
[MAE_all,MSE_all,RMSE_all,R2_all]=model_eval_unsupervised(data,all_net)
code_all = 'FC all, MSE, parameterized  ' + str(layer1_size) + ',' + str(layer2_size)
tib_net, tib_losses = train_unsupervised(tib, tib_nn, num_epochs, learn_rate, 
                                         weight_decay, batch_size, crit, p_int, device)
[MAE_tib,MSE_tib,RMSE_tib,R2_tib]=model_eval_unsupervised(tib,tib_net)
code_tib = 'FC tib, MSE, parameterized  ' + str(layer1_size) + ',' + str(layer2_size)
fem_net, fem_losses = train_unsupervised(fem, fem_nn, num_epochs, learn_rate, 
                                         weight_decay, batch_size, crit, p_int, device)
[MAE_fem,MSE_fem,RMSE_fem,R2_fem]=model_eval_unsupervised(fem,fem_net)
code_fem = 'FC fem, MSE, parameterized  ' + str(layer1_size) + ',' + str(layer2_size)

all_pred = all_net.decode(all_net.encode(torch.FloatTensor(data).cuda())).cpu().detach().numpy()
tib_pred = tib_net.decode(tib_net.encode(torch.FloatTensor(tib).cuda())).cpu().detach().numpy()
fem_pred = fem_net.decode(fem_net.encode(torch.FloatTensor(fem).cuda())).cpu().detach().numpy()
plot_network(code_all,all_net,data,all_pred,all_losses,MAE_all,RMSE_all,R2_all)
plot_network(code_tib,tib_net,tib,tib_pred,tib_losses,MAE_tib,RMSE_tib,R2_tib)
plot_network(code_fem,fem_net,fem,fem_pred,fem_losses,MAE_fem,RMSE_fem,R2_fem)

torch.save(all_net.state_dict(),'all_net'+date+'.pt')
torch.save(fem_net.state_dict(),'fem_net'+date+'.pt')
torch.save(tib_net.state_dict(),'tib_net'+date+'.pt')

#%% Train Fem Mask Parameterization network (supervised)
num_epochs = 100
learn_rate = 2e-4
weight_decay = 1e-5
batch_size = 30
p_int = num_epochs//10
seed = 1437
crit = nn.L1Loss()
inform = True

torch.manual_seed(seed)

#   mask2qct_net(prms,maxes,target_resolution,hidden=64,mp_ksize=2,mp_strd=2,
#             cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2)
fem_prm_nn = mask2qct_net(layer2_size,maxes,target_resolution,
                          64,2,2,2,1,4,16,32,0.2).cuda()
fem_nn.load_state_dict(torch.load('fem_net'+last_date+'.pt'))

if(inform):
    fem_prm_nn.load_state_dict(torch.load('fem_prm_net'+date+'.pt'))

y = fem_nn.encode(torch.FloatTensor(fem).cuda()).cpu().detach()
x_train,x_test,y_train,y_test = train_test_split(fem_mask_data.T,y,test_size=0.1,random_state=3)

fem_prm_net, losses = train_supervised(x_train, y_train, fem_prm_nn, num_epochs, learn_rate, 
                                       weight_decay, batch_size, crit, p_int, device)
[MAE_prm_fem_train,MSE_prm_fem_train,RMSE_prm_fem_train,R2_prm_fem_train,
 MAE_prm_fem,MSE_prm_fem,RMSE_prm_fem,R2_prm_fem]=model_eval_supervised(x_train,x_test,y_train,y_test,fem_prm_net)
code = 'femur' 
pred = fem_prm_net(torch.FloatTensor(x_test).cuda()).cpu().detach().numpy()
plot_network(code,fem_prm_net,y_test,pred,losses,MAE_prm_fem,RMSE_prm_fem,R2_prm_fem)

torch.save(fem_prm_net.state_dict(),'fem_prm_net'+date+'.pt')

#%% High resolution: Train fem Mask Parameterization network (supervised)

num_epochs = 100
learn_rate = 1e-4
weight_decay = 1e-5
batch_size = 50
p_int = num_epochs//10
seed=140 
crit = nn.L1Loss()
inform = False

#fem_nn = fem_net().cuda()
fem_nn.load_state_dict(torch.load('fem_net'+last_date+'.pt'))

# HR_mask2qct_net(prms,maxes,target_resolution,high_res_mag,high_res_z_reduction,
#             hidden=64,mp_ksize=8,mp_strd=4,cv_ksize=8,cv_pad=4,
#             ch_1=4,ch_2=16,ch_3=32,dropout=0.2)
    
HR_fem_prm_nn = HR_mask2qct_net(layer2_size,maxes,target_resolution,
                                high_res_mag,high_res_z_reduction,
                                64,8,4,8,4,4,16,32,0.2).cuda()

if(inform):
    HR_fem_prm_nn.load_state_dict(torch.load('HR_fem_prm_net'+date+'.pt'))

y = fem_nn.encode(torch.FloatTensor(fem).cuda()).cpu().detach()
x_train,x_test,y_train,y_test = train_test_split(HR_fem_mask_data.T,y,test_size=0.1,random_state=3)

HR_fem_prm_net, losses = train_supervised(x_train, y_train, HR_fem_prm_nn, num_epochs, learn_rate, 
                                       weight_decay, batch_size, crit, p_int, device)
[MAE_HR_prm_fem_train,MSE_HR_prm_fem_train,RMSE_HR_prm_fem_train,R2_HR_prm_fem_train,
 MAE_HR_prm_fem,MSE_HR_prm_fem,RMSE_HR_prm_fem,R2_HR_prm_fem]=model_eval_supervised(x_train,x_test,y_train,y_test,HR_fem_prm_net)
code = 'HR_femur' 
pred = HR_fem_prm_net(torch.FloatTensor(x_test).cuda()).cpu().detach().numpy()
plot_network(code,HR_fem_prm_net,y_test,pred,losses,HR_MAE_prm_fem,HR_RMSE_prm_fem,HR_R2_prm_fem)

torch.save(HR_fem_prm_net.state_dict(),'HR_fem_prm_net'+date+'.pt')


#%% Train tib Mask Parameterization network (supervised)
num_epochs = 1000
learn_rate = 2e-4
weight_decay = 1e-5
batch_size = 60
p_int = num_epochs//10
seed = 1437
crit = nn.L1Loss()
inform = False

torch.manual_seed(seed)

#   mask2qct_net(prms,maxes,target_resolution,hidden=64,mp_ksize=2,mp_strd=2,
#             cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2)
tib_prm_nn = mask2qct_net(layer2_size,maxes,target_resolution,
                          64,2,2,2,1,4,16,32,0.2).cuda()
tib_nn.load_state_dict(torch.load('tib_net'+last_date+'.pt'))

if(inform):
    tib_prm_nn.load_state_dict(torch.load('tib_prm_net'+date+'.pt'))

y = tib_nn.encode(torch.FloatTensor(tib).cuda()).cpu().detach()
x_train,x_test,y_train,y_test = train_test_split(tib_mask_data.T,y,test_size=0.1,random_state=3)

tib_prm_net, losses = train_supervised(x_train, y_train, tib_prm_nn, num_epochs, learn_rate, 
                                       weight_decay, batch_size, crit, p_int, device)
[MAE_prm_tib_train,MSE_prm_tib_train,RMSE_prm_tib_train,R2_prm_tib_train,
 MAE_prm_tib,MSE_prm_tib,RMSE_prm_tib,R2_prm_tib]=model_eval_supervised(x_train,x_test,y_train,y_test,tib_prm_net)
code = 'tibia' 
pred = tib_prm_net(torch.FloatTensor(x_test).cuda()).cpu().detach().numpy()
plot_network(code,tib_prm_net,y_test,pred,losses,MAE_prm_tib,RMSE_prm_tib,R2_prm_tib)

torch.save(tib_prm_net.state_dict(),'tib_prm_net'+date+'.pt')

#%% High resolution: Train tib Mask Parameterization network (supervised)

num_epochs = 100
learn_rate = 1e-4
weight_decay = 1e-5
batch_size = 50
p_int = num_epochs//10
seed=140 
crit = nn.L1Loss()
inform = False

#tib_nn = tib_net().cuda()
tib_nn.load_state_dict(torch.load('tib_net'+last_date+'.pt'))

# HR_mask2qct_net(prms,maxes,target_resolution,high_res_mag,high_res_z_reduction,
#             hidden=64,mp_ksize=8,mp_strd=4,cv_ksize=8,cv_pad=4,
#             ch_1=4,ch_2=16,ch_3=32,dropout=0.2)
    
HR_tib_prm_nn = HR_mask2qct_net(layer2_size,maxes,target_resolution,
                                high_res_mag,high_res_z_reduction,
                                64,8,4,8,4,4,16,32,0.2).cuda()

if(inform):
    HR_tib_prm_nn.load_state_dict(torch.load('HR_tib_prm_net'+date+'.pt'))

y = tib_nn.encode(torch.FloatTensor(tib).cuda()).cpu().detach()
x_train,x_test,y_train,y_test = train_test_split(HR_tib_mask_data.T,y,test_size=0.1,random_state=3)

HR_tib_prm_net, losses = train_supervised(x_train, y_train, HR_tib_prm_nn, num_epochs, learn_rate, 
                                       weight_decay, batch_size, crit, p_int, device)
[MAE_HR_prm_tib_train,MSE_HR_prm_tib_train,RMSE_HR_prm_tib_train,R2_HR_prm_tib_train,
 MAE_HR_prm_tib,MSE_HR_prm_tib,RMSE_HR_prm_tib,R2_HR_prm_tib]=model_eval_supervised(x_train,x_test,y_train,y_test,HR_tib_prm_net)
code = 'HR_tibia' 
pred = HR_tib_prm_net(torch.FloatTensor(x_test).cuda()).cpu().detach().numpy()
plot_network(code,HR_tib_prm_net,y_test,pred,losses,HR_MAE_prm_tib,HR_RMSE_prm_tib,HR_R2_prm_tib)

torch.save(HR_tib_prm_net.state_dict(),'HR_tib_prm_net'+date+'.pt')


#%% output
performance = np.asarray([losses,MAE_train,MSE_train,RMSE_train,R2_train,MAE,MSE,RMSE,R2])
output_file_path = args.data+args.save+args.bone+args.model+date+'.csv'
np.savetxt(output_file_path,performance,delimiter=',')

