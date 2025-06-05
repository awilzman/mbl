# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# =============================================================================

import argparse
import torch
import torch.nn as nn
import datetime
import os
import numpy as np
import arw_training_turing as tt
import dataloader as dl
import bone_networks as bn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

date = datetime.datetime.now()
month = date.strftime("%m")
day = date.strftime("%d")
year = date.strftime("%y")
date = ('_'+month + '_' + day + '_' + year)
# =============================================================================
# Parsing arguments
# =============================================================================

parser = argparse.ArgumentParser(description='PyTorch Bone Crusher')
parser.add_argument('--data', type=str, default='',
                    help='location of the data')
# =============================================================================
# data requirement: files:
#   QCT
#   args.data + args.bone + _strength.csv and mineral
#   if in code parent folder, then data can remain blank.
#   be sure to name these csvs as shown above
#   Masks
#   need to load: args.data + args.bone + '_integral.txt'
#   already pickled: args.bone+'_mask.pkl'
#   high res pickle: 'HR_'+args.bone+'_mask.pkl'
# =============================================================================

parser.add_argument('--bone', type=str, default='femur',
                    help='bone (femur, tibia)')
parser.add_argument('--model', type=str, default='M2QT',
                    help='Input2Output (EG M2Q) (M: Mask, Q: QCT Metrics, '+
                    'F: FE, G: Generator, D: Discriminator, GAN: GAN; '+
                    'Add i to end for informed version; ii for full inform'+
                    'Add HR to end for high-res mask)')
parser.add_argument('--maskxy', type=int, default=200,
                    help='max size of each mask slice [mm]')
parser.add_argument('--maskz', type=int, default=300,
                    help='max depth of scan [mm]')
parser.add_argument('--res', type=float, default=2,
                    help='mask resolution [mm/element]')
parser.add_argument('--highres', type=int, default=1,
                    help='high resolution mask magnification')
parser.add_argument('--highresred', type=float, default=0.2,
                    help='high resolution mask reduction')
parser.add_argument('--prms', type=int, default=8,
                    help='parameters')
parser.add_argument('--trunc', type=int, default=3,
                    help='informed network truncated value')
parser.add_argument('--layer1', type=int, default=20,
                    help='layer 1 size')
parser.add_argument('--hidden', type=int, default=32,
                    help='hidden layer size')
parser.add_argument('--ch1', type=int, default=4,
                    help='convolution channel 1 size')
parser.add_argument('--ch2', type=int, default=8,
                    help='convolution channel 2 size')
parser.add_argument('--ch3', type=int, default=16,
                    help='convolution channel 3 size')
parser.add_argument('--cpad', type=int, default=2,
                    help='convolution padding')
parser.add_argument('--ckernel', type=int, default=4,
                    help='convolution kernel')
parser.add_argument('--mpstride', type=int, default=4,
                    help='max pool stride')
parser.add_argument('--mpkernel', type=int, default=4,
                    help='max pool kernel')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--decay', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--games', type=int, default=5,
                    help='number of games in GAN fight')
parser.add_argument('--matches', type=int, default=2,
                    help='number of matches in GAN fight (game # multiplier)')
parser.add_argument('--seed', type=int, default=1337,
                    help='random seed')
parser.add_argument('--ttsplit', type=float, default=0.2,
                    help='train / test split')
parser.add_argument('--randstate', type=int, default=1337,
                    help='random train / test split state')
parser.add_argument('--ae_load', type=str, default='07_24_23',
                    help='autoencoder date to load (mm_dd_yy)'+
                    ' for GAN')
parser.add_argument('--load', type=str, default='no',
                    help='model date to load (mm_dd_yy)')
parser.add_argument('--save', type=str, default='model',
                    help='fun name to tag onto the saved model')
args = parser.parse_args()
# =============================================================================
# Argument errors
# =============================================================================
if len(args.model) > 5:
    raise Exception('Model name is too long, only one model at a time.')
# =============================================================================
# Load Data
# =============================================================================
# =============================================================================
# QCT DATA SETUP
#       Z:\_CODES\Ekso Clinical Trial QCT
#
#     strength_header1={'Sub' 'Trial' 'Side' 'epiLngth' 'metLngth' 'diaLngth' 
#       'extLngth' 'epiCSA' 'metCSA' 'diaCSA' 'epiCSI' 'metCSI' 'diaCSI'};
#     strength_header2={'epiBSI' 'metBSI' 'diaBSI' 'CortTotVol' 'metCTI' 
#       'epiBR' 'metBR' 'diaBR'};
#
#     mineral_header1={'Sub' 'Trial' 'Side' 'Epi_iBV' 'Epi_iBMC' 'Epi_iBMD' 
#       'Epi_tBV' 'Epi_tBMC' 'Epi_tBMD'  'Epi_cBV' 'Epi_cBMC' 'Epi_cBMD'};
#     mineral_header2={'Met_iBV' 'Met_iBMC' 'Met_iBMD' 'Met_tBV' 'Met_tBMC' 
#       'Met_tBMD'  'Met_cBV' 'Met_cBMC' 'Met_cBMD'};
#     mineral_header3={'Dia_iBV' 'Dia_iBMC' 'Dia_iBMD' 'Dia_tBV' 'Dia_tBMC' 
#       'Dia_tBMD'  'Dia_cBV' 'Dia_cBMC' 'Dia_cBMD'};
#     mineral_header4={'Tot_iBV' 'Tot_iBMC' 'Tot_iBMD' 'Tot_tBV' 'Tot_tBMC' 
#       'Tot_tBMD'  'Tot_cBV' 'Tot_cBMC' 'Tot_cBMD'};
#     mineral_header5={'Epi_eBV' 'Epi_eBMC' 'Epi_eBMD' 'Met_eBV' 'Met_eBMC' 
#       'Met_eBMD' 'Dia_eBV' 'Dia_eBMC' 'Dia_eBMD' 'Tot_eBV' 'Tot_eBMC' 'Tot_eBMD'};
#     mineral_header6={'EpiMATvol' 'MetMATvol' 'DiaMATvol'};
# 
# =============================================================================
#directory = 'C:/Users/arwilzman/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Documents/Desktop/' # for bme80
#
#%%


if 'i' in args.model or 'Q' in args.model: #load qct data if needed
# i need to concatenate current data to use here
    strength = pd.read_csv((args.data+args.bone+'_strength.csv'))
    mineral = pd.read_csv((args.data+args.bone+'_mineral.csv'))

    qct_data = np.concatenate((
        np.array(strength.iloc[:,7:]),np.array(mineral.iloc[:,3:])),axis=1)
    qct_id = np.array(strength[['Sub','Trial','Side']])
    del strength, mineral
    sz = np.size(qct_data,1)
    qct_data = np.nan_to_num(qct_data, nan=0.0)
    qct_id = pd.DataFrame(qct_id)
    qct_id = qct_id.rename({0:1,1:2,2:3},axis='columns')
    # scale each row
    qct_scaler = preprocessing.MinMaxScaler(feature_range=(0,10))
    qct_minmax = np.zeros((2,65)) #only 65 variables
    # store scaler data in t_minmax
    for i in range(65):
        qct_data[:,i] = qct_scaler.fit_transform(qct_data[:,i].reshape(-1,1))[:,0]
        qct_minmax[0,i] = float(qct_scaler.data_min_[0])
        qct_minmax[1,i] = float(qct_scaler.data_max_[0])
    # index locations for epiphysis, metaphysis, diaphysis, and total metrics
    
    epi_ind = [[0,3,6,11,14,15,16,17,18,19,20,21,22,50,51,52,62]]
    met_ind = [[1,4,7,12,23,24,25,26,27,28,29,30,31,53,54,55,63]]
    dia_ind = [[2,5,8,13,32,33,34,35,36,37,38,39,40,56,57,58,64]]
    tot_ind = [[9,10,41,42,43,44,45,46,47,48,49,59,60,61]]
    
if 'M' in args.model:
    maxes = [args.maskxy,args.maskxy,args.maskz]
    mask_folder = 'masks/'
    if 'T' in args.model:
        data_file = args.data+args.bone+'_T_mask.pkl'
        id_file = args.data+args.bone+'_T_mask_ID.pkl'
    else:
        data_file = args.data+args.bone+'_mask.pkl'
        id_file = args.data+args.bone+'_mask_ID.pkl'
        
    if 'HR' in args.model:
        if os.path.exists('HR_'+data_file):
            with open('HR_'+data_file, 'rb') as file:
                mask_data = pickle.load(file)
        else:
            if 'T' in args.model:
                mask_id, mask_data = dl.load_masks(args.data+mask_folder,args.bone+'_integral.txt',
                                                   qct_id,maxes,args.res,True,
                                                   args.highres,args.highresred)
                mask_data = np.transpose(mask_data, (2, 0, 1))
                with open('HR_'+data_file, 'wb') as f:
                    pickle.dump(mask_data, f)
                with open('HR_'+id_file, 'wb') as f:
                    pickle.dump(mask_id, f)
            else:
                mask_id, mask_data = dl.load_masks(args.data+mask_folder,args.bone+'_integral.txt',
                                                   qct_id,maxes,args.res,False,
                                                   args.highres,args.highresred)
                mean = np.mean(mask_data, axis=(0, 1, 2))
                std = np.std(mask_data, axis=(0, 1, 2))
                mask_data = (mask_data - mean) / std
                mask_data = np.transpose(mask_data, (3, 0, 1, 2))
                with open(data_file, 'wb') as f:
                    pickle.dump(mask_data, f)
                with open(id_file, 'wb') as f:
                    pickle.dump(mask_id, f)
    else:
        if os.path.exists(data_file):
            with open(data_file, 'rb') as file:
                mask_data = pickle.load(file)
        else:
            if 'T' in args.model:
                mask_id, mask_data = dl.load_masks(args.data+mask_folder,args.bone+'_integral.txt',
                                                   qct_id,maxes,args.res)
                mask_data = np.transpose(mask_data, (2, 0, 1))
                with open(data_file, 'wb') as f:
                    pickle.dump(mask_data, f)
                with open(id_file, 'wb') as f:
                    pickle.dump(mask_id, f)
            else:
                mask_id, mask_data = dl.load_masks(args.data+mask_folder,args.bone+'_integral.txt',
                                                   qct_id,maxes,args.res,False)
                mean = np.mean(mask_data, axis=(0, 1, 2))
                std = np.std(mask_data, axis=(0, 1, 2))
                mask_data = (mask_data - mean) / std
                mask_data = np.transpose(mask_data, (3, 0, 1, 2))
                with open(data_file, 'wb') as f:
                    pickle.dump(mask_data, f)
                with open(id_file, 'wb') as f:
                    pickle.dump(mask_id, f)


# =============================================================================
# Network Training
# =============================================================================
#   mask2qct_net(prms,maxes,args.res,hidden=64,mp_ksize=2,mp_strd=2,
#             cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2)
if torch.cuda.is_available():
    print('CUDA available')
    print(torch.cuda.get_device_name(0))
else:
    print('CUDA *not* available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

unsup = False
p_int = args.epochs//10
crit = nn.MSELoss()
if 'Q2Q' in args.model:
    unsup = True
    if 'i' in args.model:
        network = bn.informed_net(epi_ind, met_ind,dia_ind,tot_ind,args.trunc,
                               args.prms).cuda()
        network_code = 'i'+args.bone+'_net'
    else:
        network = bn.FC_net(args.layer1,args.prms,sz).cuda()
        network_code = args.bone+'_net'
    
if 'M2' in args.model:
    maxes = [args.maskxy,args.maskxy,args.maskz]
    if 'T' in args.model:
        network = bn.mask_transformer(mask_data.shape[2],
                                      mask_data.shape[2],args.hidden,
                                      args.prms).cuda()
        pos_network = bn.mask_position(mask_data.shape[1],args.dropout).cuda()
    else:
        network = bn.mask2qct_net(args.prms,maxes,args.res,args.hidden,
                                  args.mpkernel,args.mpstride,args.ckernel,
                                  args.cpad,args.ch1,args.ch2,args.ch3,
                                  args.dropout).cuda()
    if 'i' in args.model:
        network_code = 'i'+args.bone+'_net'
        encoder_network = bn.informed_net(epi_ind, met_ind,dia_ind,tot_ind,
                                          args.trunc,args.prms).cuda()
    else:
        network_code = args.bone+'_net'
        encoder_network = bn.FC_net(args.layer1,args.prms,sz).cuda()
        
    encoder_network.load_state_dict(torch.load(network_code+date+'.pt'))
    
    
    if 'T' in args.model:
        network_code = 'T_' + network_code
    y = encoder_network.encode(torch.FloatTensor(qct_data).cuda()).cpu().detach()
    
# Load FE data and reduce masks 
if 'F' in args.model:
    fe_data = pd.read_csv(args.data+'fe_data.csv', sep=',', header=None)
    fe_data[[3]]+=1 # left right notation needs to match, my bad
    cols = [1,2,3]
    data_id = pd.DataFrame(qct_id)
    fe_data = fe_data.merge(qct_id, on=cols, how='left')
    fe_data = fe_data.dropna() # find matching indices, drop missing
    matched = fe_data.index
    if 'T' in args.model:
        mask_data = mask_data[matched,:,:] # remove masks without FE
    else:
        mask_data = mask_data[matched,:,:,:] # remove masks without FE
    y = fe_data.iloc[:,0].values
    fe_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
    fe_minmax = np.zeros(2)
    # store scaler data in t_minmax
    fe_data = fe_data.values[:,0].reshape(-1,1)
    fe_data = fe_scaler.fit_transform(fe_data)
    fe_minmax[0] = float(fe_scaler.data_min_[0])
    fe_minmax[1] = float(fe_scaler.data_max_[0])
    
if 'M2F' in args.model:
    network_code = args.bone+'_fe_net'
    if 'T' in args.model:
        network_code = 'T_' + network_code
    else:
        network = bn.mask2fe_net(args.prms,maxes,args.res,args.hidden,
                                 args.mpkernel,args.mpstride,args.ckernel,
                                 args.cpad,args.ch1,args.ch2,args.ch3,
                                 args.dropout).cuda()
        if 'i' in args.model:
            network_code = 'i' + network_code
            if 'ii' in args.model:
                network_code = 'i' + network_code
                prm_net = args.bone+'_net'
                encoder_network = bn.FC_net(args.layer1,args.prms,sz).cuda()
                encoder_network.load_state_dict(torch.load(prm_net+date+'.pt'))
                qct_data = encoder_network.encode(torch.FloatTensor(data).cuda())
                #reduce qct data to matched
                qct_data = qct_data[matched,:]
            else:
                prm_net = args.bone+'_mask2qct_net'
                encoder_network = bn.mask2qct_net(args.prms,maxes,args.res,args.hidden,
                                                  args.mpkernel,args.mpstride,args.ckernel,
                                                  args.cpad,args.ch1,args.ch2,args.ch3,
                                                  args.dropout).cuda()
                encoder_network.load_state_dict(torch.load(prm_net+date+'.pt'))
                with torch.no_grad():
                    qct_data = encoder_network(torch.FloatTensor(mask_data).cuda())
                    # mask data is already matched to fe
        
# =============================================================================
#     GAN code
#       Requries finished autoencoder network to get started
#       Optional load for discriminator and generator, but not 'either or'
#           must be same date for now
# =============================================================================
if 'GAN' in args.model:
    netD = bn.netD(args.prms,args.layer1,args.hidden,sz).cuda()
    netG = bn.netG(args.prms,args.layer1,args.hidden,sz).cuda()
    if 'i' in args.model:
        autoencoder_network = bn.informed_net(epi_ind, met_ind,dia_ind,tot_ind,
                                       args.trunc,args.prms).cuda()
    else: 
        autoencoder_network = bn.FC_net(args.layer1,args.prms,sz).cuda()
    if 'i' in args.model:
        if 'ii' in args.model:
            network_code = 'ii' + args.bone
        else:
            network_code = 'i' + args.bone
    else:
        network_code = args.bone + '_net'

    autoencoder_network.load_state_dict(torch.load(network_code+
                                                   '_'+args.ae_load+'.pt'))   
    
    if '_' in args.load: # if load previous model
        d_name = 'd_'+network_code+'_'+args.load+'.pt'
        g_name = 'g_'+network_code+'_'+args.load+'.pt'
        netD.load_state_dict(torch.load(args.data+d_name))
        netG.load_state_dict(torch.load(args.data+d_name))
    
    noise = torch.randn((qct_data.shape[0],args.prms),device=device)
    noise_dec = autoencoder_network.decode(noise)
    noise_lab = torch.cat((noise_dec,torch.zeros((qct_data.shape[0],1)).cuda()),1)
    real_data = autoencoder_network.encode(torch.FloatTensor(qct_data).cuda())
    real_data_lab = torch.cat((autoencoder_network.decode(
        real_data),torch.ones((data.shape[0],1)).cuda()),1)
    mix_data = torch.cat((noise_lab,real_data_lab),0).cpu().detach().numpy()
    x = torch.FloatTensor(mix_data[:,0:mix_data.shape[1]-1]).cuda(
        ).cpu().detach().numpy()
    y = mix_data[:,-1].reshape((mix_data.shape[0],1))
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=args.ttsplit,random_state=args.randstate)
    d_net, losses = tt.train_supervised(
        x_train, y_train, netD, args.epochs, args.lr, args.batch_size, crit, p_int)
    [MAE_train,MSE_train,RMSE_train,R2_train,
     MAE,MSE,RMSE,R2]=tt.model_eval_supervised(x_train,x_test,y_train,y_test,d_net)
    x_real = autoencoder_network.decode(real_data).cpu().detach().numpy()
    g_net, g_net_losses, d_net, d_net_losses = tt.train_GD(
        netG, d_net,x_real, args.epochs, args.lr, args.batch_size, crit, p_int,device)
    d_net,g_net,MAE,MSE,RMSE,R2,g_net_losses, d_net_losses=tt.GAN_Fight(
        d_net,g_net,args.matches,args.games,args.epochs,args.lr,args.batch_size,p_int,device)
    torch.save(d_net.state_dict(),args.data+'d_'+network_code+'_'+date+'.pt')
    torch.save(g_net.state_dict(),args.data+'g_'++network_code+'_'+date+'.pt')
    
else: #if not GAN
    if '_' in args.load: # if load previous model
        last_date = '_'+args.load
        network_name = network_code+last_date+'.pt'
        network.load_state_dict(torch.load(network_name))
# =============================================================================
#     Unsupervised models
#       these must have .encode and .decode modules within them
# =============================================================================
    if unsup:
        network, losses = tt.train_unsupervised(qct_data, network, args.epochs, 
                                             args.lr,args.decay,args.batch_size,
                                             crit, p_int, device)
        [MAE,MSE,RMSE,R2]=tt.model_eval_unsupervised(qct_data,network)
        x_test = qct_data
        y_test = qct_data
# =============================================================================
#     Supervised models
# =============================================================================
    else:
        if 'M' in args.model:
            if 'i' in args.model:
                if 'T' in args.model:
                    mask_data = mask_data #informed tranformer?
                else:
                    qct_data = qct_data.cpu().detach()
                    mask_data[:, -8:, -100:, -200:] = qct_data[:, :, np.newaxis, np.newaxis]
                    
            x_train,x_test,y_train,y_test = train_test_split(mask_data,y,
                                                             test_size=args.ttsplit,
                                                             random_state=args.randstate)
        network, losses = tt.train_supervised(x_train, y_train, network, args.epochs,
                                              args.lr, args.decay, args.batch_size,
                                              crit, p_int, device)
        [MAE_train,MSE_train,RMSE_train,R2_train,MAE,MSE,RMSE,R2
         ]=tt.model_eval_supervised(x_train,x_test,y_train,y_test,network)
        
    network_name = network_code+date+'.pt'
    torch.save(network.state_dict(),network_name)
    network.eval()
    x_test = torch.FloatTensor(x_test).to(device)
    y_pred = network(x_test)
    y_pred = y_pred.cpu().detach().numpy().flatten()
    y_test = y_test.flatten()
    testing_data = np.concatenate((y_pred,y_test))
    # Uncomment next line to see test data
    #np.savetxt((args.save+args.bone+args.model+date+'_test'+'.csv'),testing_data,delimiter=',')

arguments = np.asarray([args.maskxy,args.maskz,args.res,args.highres,
                        args.highresred,args.prms,args.trunc,args.layer1,
                        args.hidden,args.ch1,args.ch2,args.ch3,args.cpad,
                        args.mpstride,args.mpkernel,args.lr,args.decay,
                        args.epochs,args.batch_size,args.dropout])

if 'GAN' in args.model:
    performance = np.asarray([MAE_train,MSE_train,RMSE_train,R2_train,
                              MAE,MSE,RMSE,R2])
else:
    if unsup:
        performance = np.asarray([MAE,MSE,RMSE,R2])
    else:
        performance = np.asarray([MAE_train,MSE_train,RMSE_train,R2_train,
                                  MAE,MSE,RMSE,R2])
output_arg_file_path = 'args_'+args.save+args.bone+args.model+date+'.csv'
np.savetxt(output_arg_file_path,arguments,delimiter=',')
output_file_path = args.save+args.bone+args.model+date+'.csv'
np.savetxt(output_file_path,performance,delimiter=',')