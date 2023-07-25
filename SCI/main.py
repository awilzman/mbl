# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# =============================================================================

import argparse
import os
import torch
import torch.nn as nn
import datetime

import arw_training
import SCI_dataloader
import bone_networks

date = datetime.datetime.now()
month = date.strftime("%m")
day = date.strftime("%d")
year = date.strftime("%y")
date = ('_'+month + '_' + day + '_' + year)

# =============================================================================
# Parsing arguments
# =============================================================================
parser = argparse.ArgumentParser(description='PyTorch Bone Crusher')
parser.add_argument('--data', type=str, default='./',
                    help='location of the data')
parser.add_argument('--bone', type=str, default='femur',
                    help='bone (femur, tibia)')
parser.add_argument('--model', type=str, default='M2Q',
                    help='Input2Output (EG M2Q) (M: Mask, Q: QCT Metrics, '+
                    'F: FE, G: Generator, D: Discriminator, GAN: GAN; '+
                    'Add i to end for informed version; '+
                    'Add HR to end for high-res mask)')
parser.add_argument('--maskxy', type=int, default=200,
                    help='max size of each mask slice [mm]')
parser.add_argument('--maskz', type=int, default=300,
                    help='max depth of scan [mm]')
parser.add_argument('--res', type=float, default=4,
                    help='mask resolution [mm/element]')
parser.add_argument('--highres', type=int, default=4,
                    help='high resolution mask magnification')
parser.add_argument('--highresred', type=float, default=0.2,
                    help='high resolution mask reduction')
parser.add_argument('--prms', type=int, default=8,
                    help='parameters', default=8)
parser.add_argument('--trunc', type=int, default=3,
                    help='informed network truncated value')
parser.add_argument('--layer1', type=int, default=40,
                    help='layer 1 size')
parser.add_argument('--hidden', type=int, default=64,
                    help='hidden layer size')
parser.add_argument('--ch1', type=int, default=4,
                    help='convolution channel 1 size')
parser.add_argument('--ch2', type=int, default=16,
                    help='convolution channel 2 size')
parser.add_argument('--ch3', type=int, default=32,
                    help='convolution channel 3 size')
parser.add_argument('--cstride', type=int, default=2,
                    help='convolution stride')
parser.add_argument('--cpad', type=int, default=2,
                    help='convolution kernel')
parser.add_argument('--mpstride', type=int, default=2,
                    help='max pool stride')
parser.add_argument('--mpkernel', type=int, default=2,
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
# Load QCT Data
# =============================================================================

studies = ['Feb28_SCI_rowing','Ekso','Healthy']

if torch.cuda.is_available():
    print('CUDA available')
    print(torch.cuda.get_device_name(0))
else:
    print('CUDA *not* available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
[tib, tib_id, fem, fem_id] = load_qct(args.data+'QCT/',studies)
sz = np.size(fem,1) 
# load correct bone data
if args.bone == 'tibia':
    tib[np.isnan(tib)] = 0
    data = tib
    data_id = tib_id
    del fem, fem_id, tib, tib_id
if args.bone == 'femur':
    fem[np.isnan(fem)] = 0
    data = fem
    data_id = fem_id
    del fem, fem_id, tib, tib_id
# scale each row
scaler = preprocessing.MinMaxScaler(feature_range=(0,10))
t_minmax = np.zeros((2,65)) #only 65 variables
# store scaler data in t_minmax
for i in range(65):
    data[:,i] = scaler.fit_transform(data[:,i].reshape(-1,1))[:,0]
    t_minmax[0,i] = scaler.data_min_
    t_minmax[1,i] = scaler.data_max_
# index locations for epiphysis, metaphysis, diaphysis, and total metrics
epi_ind = [[0,3,6,11,14,15,16,17,18,19,20,21,22,50,51,52,62]]
met_ind = [[1,4,7,12,23,24,25,26,27,28,29,30,31,53,54,55,63]]
dia_ind = [[2,5,8,13,32,33,34,35,36,37,38,39,40,56,57,58,64]]
tot_ind = [[9,10,41,42,43,44,45,46,47,48,49,59,60,61]]
# =============================================================================
# Load Mask Data
# =============================================================================
if 'M' in args.model:
    if args.bone == 'femur':
        mask_data, HR_mask_data = load_masks(args.data+'masks/',
                                             '_fm_integral.txt',data_id,maxes,
                                             target_resolution,high_res_mag,
                                             high_res_z_reduction)
    if args.bone == 'tibia':
        mask_data, HR_mask_data = load_masks(args.data+'masks/',
                                             '_tb_integral.txt',data_id,maxes,
                                             target_resolution,high_res_mag,
                                             high_res_z_reduction)
# =============================================================================
# Network Training
# =============================================================================
#   mask2qct_net(prms,maxes,target_resolution,hidden=64,mp_ksize=2,mp_strd=2,
#             cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2)
unsup = False
p_int = args.epochs//10
crit = nn.L1Loss()
if args.model == 'Q2Q':
    unsup = True
    network = FC_net(args.layer1,args.prms,sz).cuda()
    if args.bone == 'tibia':
        network_code = 'tib_net'
    if args.bone == 'femur':
        network_code = 'fem_net'
if args.model == 'Q2Qi':
    unsup = True
    network = informed_net(epi_ind, met_ind,dia_ind,tot_ind,args.trunc,
                           args.prms).cuda()
    if args.bone == 'tibia':
        network_code = 'itib_net'
    if args.bone == 'femur':
        network_code = 'ifem_net'
if 'M2Q' in args.model:
    maxes = [args.maskxy,args.maskxy,args.maskz]
    if 'i' in args.model:
        encoder_network = informed_net(epi_ind, met_ind,dia_ind,tot_ind,
                                       args.trunc,args.prms).cuda()
    else:
        encoder_network = FC_net(args.layer1,args.prms,sz).cuda()
    if 'HR' in args.model:
        mask_data = HR_mask_data
        network = HR_mask2qct_net(args.prms,maxes,args.res,args.highres,
                                  args.highresred,args.hidden,args.mpkernel,
                                  args.mpstride,args.ckernel,args.cpad,
                                  args.ch1,args.ch2,args.ch3,args.dropout).cuda()
    else:
        network = mask2qct_net(args.prms,maxes,args.res,args.hidden,
                               args.mpkernel,args.mpstride,args.ckernel,
                               args.cpad,args.ch1,args.ch2,args.ch3,
                               args.dropout).cuda()
    del HR_mask_data
    if args.bone == 'tibia':
        network_code = 'tib_prm_net'
        encoder_network.load_state_dict(torch.load('tib_net'+date+'.pt'))
    if args.bone == 'femur':
        network_code = 'fem_prm_net'
        encoder_network.load_state_dict(torch.load('fem_net'+date+'.pt'))
    y = encoder_network.encode(torch.FloatTensor(data).cuda()).cpu().detach()
    data = mask_data
    del mask_data
# =============================================================================
#     GAN code
# =============================================================================
if 'GAN' in args.model:
    netD = netD(args.prms,args.layer1,args.hidden,sz).cuda()
    netG = netG(args.prms,args.layer1,args.hidden,sz).cuda()
    autoencoder_network = FC_net(args.layer1,args.prms,sz).cuda()
    
    if args.bone == 'tibia':
        network_code = 'tib'
    if args.bone == 'femur':
        network_code = 'fem'
    autoencoder_network.load_state_dict(torch.load(network_code+
                                                   '_'+args.ae_load))    
    if '_' in args.load: # if load previous model
        d_name = 'd_'+network_code+'_'+args.load+'.pt'
        g_name = 'g_'+network_code+'_'+args.load+'.pt'
        netD.load_state_dict(torch.load(d_name))
        netG.load_state_dict(torch.load(g_name))
    
    noise = torch.randn((data.shape[0],args.prms),device=device)
    noise_dec = autoencoder_network.decode(noise)
    noise_lab = torch.cat((noise_dec,torch.zeros((data.shape[0],1)).cuda()),1)
    real_data = autoencoder_network.encode(torch.FloatTensor(data).cuda())
    real_data_lab = torch.cat((autoencoder_network.decode(
        real_data),torch.ones((data.shape[0],1)).cuda()),1)
    mix_data = torch.cat((noise_lab,real_data_lab),0).cpu().detach().numpy()
    x = torch.FloatTensor(mix_data[:,0:mix_data.shape[1]-1]).cuda(
        ).cpu().detach().numpy()
    y = mix_data[:,-1].reshape((mix_data.shape[0],1))
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=args.ttsplit,random_state=args.randstate)
    d_net, losses = train_sup(
        x_train, y_train, netD, args.epochs, args.lr, args.batch_size, crit, p_int)
    plt.plot(losses)
    plt.title('losses')
    plt.show()
    [MAE_train,MSE_train,RMSE_train,R2_train,
     MAE,MSE,RMSE,R2]=model_eval_supervised(x_train,x_test,y_train,y_test,d_net)
    x_real = autoencoder_network.decode(real_data).cpu().detach().numpy()
    g_net, g_net_losses, d_net, d_net_losses = train_GD(
        x_real, netG, d_net, args.epochs, args.lr, args.batch_size, crit, p_int)
    d_net,g_net,MAE,MSE,RMSE,R2=GAN_Fight(d_net,g_net,args.matches,args.games,
                          args.epochs,args.lr,args.batch_size)
    torch.save(d_net.state_dict(),'d_'+network_code+'_'+date+'.pt')
    torch.save(g_net.state_dict(),'g_'++network_code+'_'+date+'.pt')
    
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
        network, losses = train_unsupervised(data, network, args.epochs, 
                                             args.lr,args.decay,args.batch_size,
                                             crit, p_int, device)
        [MAE,MSE,RMSE,R2]=model_eval_unsupervised(data,network)
# =============================================================================
#     Supervised models
# =============================================================================
    else:
        x_train,x_test,y_train,y_test = train_test_split(data.T,y,
                                                         test_size=args.ttsplit,
                                                         random_state=args.rstate)
        network, losses = train_supervised(x_train, y_train, network, args.epochs, 
                                           args.lr, args.decay, args.batch_size, 
                                           crit, p_int, device)
        [MAE_train,MSE_train,RMSE_train,R2_train,MAE,MSE,RMSE,R2
         ]=model_eval_supervised(x_train,x_test,y_train,y_test,network)
        
    network_name = network_code+date+'.pt'
    torch.save(network.state_dict(),network_name)
return MAE,MSE,RMSE,R2
