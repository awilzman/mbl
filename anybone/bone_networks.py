# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# sz needs to be defined as data shape[1]
# networks:
#   informed_net:
#       QCT metrics autoencoder, metric location-informed  
#           see eg data Z:\_Current IRB Approved Studies\IRB_Closed\
#                       FES_Rowing_SecondaryAnalysis\Final QCT Results
#   FC_net:
#       QCT metrics autoencoder, Fully Connected
#   mask2qct_net:
#       Mask 3D density matrix to QCT metrics reduced by encoder of above 
#   HR_mask2qct_net:
#       Same, but with higher resolution data, truncated view
#   netG:
#       Generator for GAN
#   netD:
#       Discriminator for GAN
# =============================================================================
import pandas as pd
import numpy as np
import torch
from torch import nn
import math


class informed_net(nn.Module):
    def __init__(self,epi_ind,met_ind,dia_ind,tot_ind,trunc,prms):
        super().__init__()
        self.epi_ind = torch.tensor(epi_ind).cuda()
        self.met_ind = torch.tensor(met_ind).cuda()
        self.dia_ind = torch.tensor(dia_ind).cuda()
        self.tot_ind = torch.tensor(tot_ind).cuda()
        self.trunc = trunc
# =============================================================================
#       slayer: works with epi/met/dia data. Each index matches to similar metrics
#       for each. Layers are separated to learn how each differ between people 
#       separately.
# =============================================================================
        self.slayer_epi_1 = nn.Linear(self.epi_ind.shape[1],round(self.epi_ind.shape[1]/2))
        self.slayer_epi_2 = nn.Linear(round(self.epi_ind.shape[1]/2),trunc)
        self.slayer_met_1 = nn.Linear(self.met_ind.shape[1],round(self.met_ind.shape[1]/2))
        self.slayer_met_2 = nn.Linear(round(self.met_ind.shape[1]/2),trunc)
        self.slayer_dia_1 = nn.Linear(self.dia_ind.shape[1],round(self.dia_ind.shape[1]/2))
        self.slayer_dia_2 = nn.Linear(round(self.dia_ind.shape[1]/2),trunc)
        
        self.slayer_tot_1 = nn.Linear(self.tot_ind.shape[1],round(self.tot_ind.shape[1]/2))
        self.slayer_tot_2 = nn.Linear(round(self.tot_ind.shape[1]/2),trunc)
        
        self.param = nn.Linear(trunc*4,prms)
        self.re_param = nn.Linear(prms,trunc*4)
        
        self.reslayer_epi_1 = nn.Linear(trunc,round(self.epi_ind.shape[1]/2))
        self.reslayer_epi_2 = nn.Linear(round(self.epi_ind.shape[1]/2),self.epi_ind.shape[1])
        self.reslayer_met_1 = nn.Linear(trunc,round(self.met_ind.shape[1]/2))
        self.reslayer_met_2 = nn.Linear(round(self.met_ind.shape[1]/2),self.met_ind.shape[1])
        self.reslayer_dia_1 = nn.Linear(trunc,round(self.dia_ind.shape[1]/2))
        self.reslayer_dia_2 = nn.Linear(round(self.dia_ind.shape[1]/2),self.dia_ind.shape[1])
        
        self.reslayer_tot_1 = nn.Linear(trunc,round(self.tot_ind.shape[1]/2))
        self.reslayer_tot_2 = nn.Linear(round(self.tot_ind.shape[1]/2),self.tot_ind.shape[1])
        
        # ReLU to disregard negative values
        self.activate = nn.ReLU()
        
    def encode(self, x):
        epi = self.activate(self.slayer_epi_1(torch.index_select(x,dim=1,index=self.epi_ind.squeeze())))
        met = self.activate(self.slayer_met_1(torch.index_select(x,dim=1,index=self.met_ind.squeeze())))
        dia = self.activate(self.slayer_dia_1(torch.index_select(x,dim=1,index=self.dia_ind.squeeze())))
        tot = self.activate(self.slayer_tot_1(torch.index_select(x,dim=1,index=self.tot_ind.squeeze())))
        
        epi = self.activate(self.slayer_epi_2(epi))
        met = self.activate(self.slayer_met_2(met))
        dia = self.activate(self.slayer_dia_2(dia))
        tot = self.activate(self.slayer_tot_2(tot))
        
        x = self.activate(self.param(torch.cat((epi,met,dia,tot),1)))
        
        return x
    
    def decode(self, x):
        
        y = torch.zeros(x.shape[0],self.epi_ind.shape[1]+self.met_ind.shape[1]+self.dia_ind.shape[1]+self.tot_ind.shape[1]).cuda()
        x = self.activate(self.re_param(x))
        
        epi = torch.index_select(x,dim=1,index=torch.tensor(range(self.trunc)).cuda())
        met = torch.index_select(x,dim=1,index=torch.tensor(range(self.trunc,self.trunc*2)).cuda())
        dia = torch.index_select(x,dim=1,index=torch.tensor(range(self.trunc*2,self.trunc*3)).cuda())
        tot = torch.index_select(x,dim=1,index=torch.tensor(range(self.trunc*3,self.trunc*4)).cuda())
        
        epi = self.activate(self.reslayer_epi_2(self.activate(self.reslayer_epi_1(epi))))
        met = self.activate(self.reslayer_met_2(self.activate(self.reslayer_met_1(met))))
        dia = self.activate(self.reslayer_dia_2(self.activate(self.reslayer_dia_1(dia))))
        tot = self.activate(self.reslayer_tot_2(self.activate(self.reslayer_tot_1(tot))))
    
        y[:,self.epi_ind.squeeze()] = epi
        y[:,self.met_ind.squeeze()] = met
        y[:,self.dia_ind.squeeze()] = dia
        y[:,self.tot_ind.squeeze()] = tot
        
        return y
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# Dense network, naive
class FC_net(nn.Module):
    def __init__(self,layer1_size,prms,sz):
        super().__init__()
        
        self.lay1 = nn.Linear(sz,layer1_size)
        self.lay2 = nn.Linear(layer1_size,prms)
        self.lay3 = nn.Linear(prms,layer1_size)
        self.lay4 = nn.Linear(layer1_size,sz)
        
        
        self.activate = nn.ReLU()
    def encode(self, x):
        h1 = self.activate(self.lay1(x))
        return self.activate(self.lay2(h1))
    def decode(self, x):
        h2 = self.activate(self.lay3(x))
        return self.activate(self.lay4(h2))
    def forward(self, x):
        x = self.encode(x)
        y = self.decode(x)
        return y

class mixed_net(nn.Module):
    def __init__(self,layer1_size,prms,trunc,sz,epi_ind,met_ind,dia_ind,tot_ind):
        super().__init__()
        self.activate = nn.ReLU()
        # FC NET
        self.lay1 = nn.Linear(sz,layer1_size)
        self.lay2 = nn.Linear(layer1_size,prms)
        self.lay3 = nn.Linear(prms,layer1_size)
        self.lay4 = nn.Linear(layer1_size,sz)
        
        # INFORMED NET
        self.epi_ind = torch.tensor(epi_ind).cuda()
        self.met_ind = torch.tensor(met_ind).cuda()
        self.dia_ind = torch.tensor(dia_ind).cuda()
        self.tot_ind = torch.tensor(tot_ind).cuda()
        self.trunc = trunc
        self.slayer_epi_1 = nn.Linear(self.epi_ind.shape[1],round(self.epi_ind.shape[1]/2))
        self.slayer_epi_2 = nn.Linear(round(self.epi_ind.shape[1]/2),trunc)
        self.slayer_met_1 = nn.Linear(self.met_ind.shape[1],round(self.met_ind.shape[1]/2))
        self.slayer_met_2 = nn.Linear(round(self.met_ind.shape[1]/2),trunc)
        self.slayer_dia_1 = nn.Linear(self.dia_ind.shape[1],round(self.dia_ind.shape[1]/2))
        self.slayer_dia_2 = nn.Linear(round(self.dia_ind.shape[1]/2),trunc)
        
        self.slayer_tot_1 = nn.Linear(self.tot_ind.shape[1],round(self.tot_ind.shape[1]/2))
        self.slayer_tot_2 = nn.Linear(round(self.tot_ind.shape[1]/2),trunc)
        
        self.param = nn.Linear(trunc*4,prms)
        self.crosstalk = nn.Linear(prms*2,prms)
        self.re_crosstalk = nn.Linear(prms,prms*2)
        self.re_param = nn.Linear(prms,trunc*4)
        
        self.reslayer_epi_1 = nn.Linear(trunc,round(self.epi_ind.shape[1]/2))
        self.reslayer_epi_2 = nn.Linear(round(self.epi_ind.shape[1]/2),self.epi_ind.shape[1])
        self.reslayer_met_1 = nn.Linear(trunc,round(self.met_ind.shape[1]/2))
        self.reslayer_met_2 = nn.Linear(round(self.met_ind.shape[1]/2),self.met_ind.shape[1])
        self.reslayer_dia_1 = nn.Linear(trunc,round(self.dia_ind.shape[1]/2))
        self.reslayer_dia_2 = nn.Linear(round(self.dia_ind.shape[1]/2),self.dia_ind.shape[1])
        
        self.reslayer_tot_1 = nn.Linear(trunc,round(self.tot_ind.shape[1]/2))
        self.reslayer_tot_2 = nn.Linear(round(self.tot_ind.shape[1]/2),self.tot_ind.shape[1])

    
    def encode(self, x):
        # FC NET
        h1 = self.activate(self.lay1(x))
        h1 = self.activate(self.lay2(h1))
        # INFORMED NET
        epi = self.activate(self.slayer_epi_1(torch.index_select(x,dim=1,index=self.epi_ind.squeeze())))
        met = self.activate(self.slayer_met_1(torch.index_select(x,dim=1,index=self.met_ind.squeeze())))
        dia = self.activate(self.slayer_dia_1(torch.index_select(x,dim=1,index=self.dia_ind.squeeze())))
        tot = self.activate(self.slayer_tot_1(torch.index_select(x,dim=1,index=self.tot_ind.squeeze())))
        
        epi = self.activate(self.slayer_epi_2(epi))
        met = self.activate(self.slayer_met_2(met))
        dia = self.activate(self.slayer_dia_2(dia))
        tot = self.activate(self.slayer_tot_2(tot))
        
        x = self.activate(self.param(torch.cat((epi,met,dia,tot),1)))
        
        encoded = self.crosstalk(torch.cat((h1,x),1))
        return encoded
    
    def decode(self, x):
        x = self.re_crosstalk(x)
        #FC NET
        h2 = self.activate(self.lay3(x[:,:x.shape[1]//2]))
        h2 = self.activate(self.lay4(h2))
        #INFORMED NET
        h1 = x[:,(x.shape[1]//2):]
        y = torch.zeros(x.shape[0],self.epi_ind.shape[1]+self.met_ind.shape[1]+self.dia_ind.shape[1]+self.tot_ind.shape[1]).cuda()
        h1 = self.activate(self.re_param(h1))
        
        epi = torch.index_select(h1,dim=1,index=torch.tensor(range(self.trunc)).cuda())
        met = torch.index_select(h1,dim=1,index=torch.tensor(range(self.trunc,self.trunc*2)).cuda())
        dia = torch.index_select(h1,dim=1,index=torch.tensor(range(self.trunc*2,self.trunc*3)).cuda())
        tot = torch.index_select(h1,dim=1,index=torch.tensor(range(self.trunc*3,self.trunc*4)).cuda())
        
        epi = self.activate(self.reslayer_epi_2(self.activate(self.reslayer_epi_1(epi))))
        met = self.activate(self.reslayer_met_2(self.activate(self.reslayer_met_1(met))))
        dia = self.activate(self.reslayer_dia_2(self.activate(self.reslayer_dia_1(dia))))
        tot = self.activate(self.reslayer_tot_2(self.activate(self.reslayer_tot_1(tot))))
    
        y[:,self.epi_ind.squeeze()] = epi
        y[:,self.met_ind.squeeze()] = met
        y[:,self.dia_ind.squeeze()] = dia
        y[:,self.tot_ind.squeeze()] = tot
        
        decoded = (h2+y)/2.0
        
        return decoded
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
        
class mask2qct_net(nn.Module):
    def __init__(self,prms,maxes,target_resolution,high_res_mag=1,
                 high_res_z_reduction=1,hidden=64,mp_ksize=2,mp_strd=2,
                 cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2):
        super().__init__()
        self.target_resolution = target_resolution
        self.high_res_mag = high_res_mag
        self.activate = nn.ReLU()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=ch_1, kernel_size=cv_ksize, padding=cv_pad),
            nn.BatchNorm3d(ch_1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),

            nn.Conv3d(in_channels=ch_1, out_channels=ch_2, kernel_size=cv_ksize, padding=cv_pad),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),

            nn.Conv3d(in_channels=ch_2, out_channels=ch_3, kernel_size=cv_ksize, padding=cv_pad),
            nn.BatchNorm3d(ch_3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd)
        )
        self.fc_input_size = self._calculate_fc_input_size(maxes)
        self.fc1 = nn.Linear(self.fc_input_size, hidden)  # Calculate the size after 3D convolutions
        self.fc2 = nn.Linear(hidden, prms)
        
    def _calculate_fc_input_size(self,maxes):
        # Dummy input to calculate feature map size after the convolutions
        if self.high_res_mag > 1:
            dummy_input = torch.rand(1, 1, int(self.high_res_mag*maxes[0]//target_resolution), 
                                     int(self.high_res_mag*maxes[1]//target_resolution), 
                                     int(high_res_z_reduction*self.high_res_mag*maxes[2]//target_resolution))
        else: 
            dummy_input = torch.rand(1, 1, int(maxes[0]//self.target_resolution), 
                                     int(maxes[1]//self.target_resolution), 
                                     int(maxes[2]//self.target_resolution))
        x = self.conv_layers(dummy_input)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1,self.fc_input_size)
        x = self.fc2(self.activate(self.fc1(x)))
        return x
    
class HR_mask2qct_net(nn.Module):
    def __init__(self,prms,maxes,target_resolution,high_res_mag,high_res_z_reduction,
                hidden=64,mp_ksize=8,mp_strd=4,cv_ksize=8,cv_pad=4,
                ch_1=4,ch_2=16,ch_3=32,dropout=0.2):
        super().__init__()
        self.high_res_mag = high_res_mag
        self.activate = nn.ReLU()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=ch_1, kernel_size=cv_ksize, padding=cv_pad),
            nn.BatchNorm3d(ch_1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),

            nn.Conv3d(in_channels=ch_1, out_channels=ch_2, kernel_size=cv_ksize, padding=cv_pad),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),
            
            nn.Conv3d(in_channels=ch_2, out_channels=ch_3, kernel_size=cv_ksize, padding=cv_pad),
            nn.BatchNorm3d(ch_3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),
        )
        self.fc_input_size = self._calculate_fc_input_size(maxes)
        self.fc1 = nn.Linear(self.fc_input_size, hidden)  # Calculate the size after 3D convolutions
        self.fc2 = nn.Linear(hidden, prms)
    def _calculate_fc_input_size(self,maxes):
        dummy_input = torch.rand(1, 1, int(self.high_res_mag*maxes[0]//target_resolution), 
                                 int(self.high_res_mag*maxes[1]//target_resolution), 
                                 int(high_res_z_reduction*self.high_res_mag*maxes[2]//target_resolution))
        x = self.conv_layers(dummy_input)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1,self.fc_input_size)
        x = self.fc2(self.activate(self.fc1(x)))
        return x
    
class netG(nn.Module):
    def __init__(self,h1,h2,h3,h4):
        super().__init__()
        self.p = nn.Linear(h1,h1)
        self.d1 = nn.Linear(h1,h2)
        self.d2 = nn.Linear(h2,h3)
        self.d3 = nn.Linear(h3,h4)
        self.activate = nn.LeakyReLU()
        
    def decode(self,x):
        x = self.activate(self.p(x))
        x = self.activate(self.d1(x))
        x = self.activate(self.d2(x))
        x = self.activate(self.d3(x))
        return x
    def forward(self,x):
        x = self.decode(x)
        return x
    
class netD(nn.Module):
    def __init__(self,h1,h2,h3,h4):
        super().__init__()
        
        self.determine1 = nn.Linear(h2,h1)
        self.determine2 = nn.Linear(h1,1)
        self.lay1 = nn.Linear(h4,h3)
        self.lay2 = nn.Linear(h3,h2)
        self.activate = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        
    def reg_encode(self, x):
        x = self.activate(self.lay1(x))
        return self.activate(self.lay2(x))
        
    def forward(self,x):
        x = self.reg_encode(x)
        x = self.sig(self.determine2(self.activate(self.determine1(x))))
        return c
  
# =============================================================================
# Mask Networks    
#   
# The first two are convolutional while the following
#   use a transformer
#     
#   
# =============================================================================
class mask2fe_net(nn.Module):
    def __init__(self,prms,maxes,target_resolution,hidden=64,mp_ksize=2,mp_strd=2,
                 cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2):
        super().__init__()
        self.activate = nn.ReLU()
        self.target_resolution = target_resolution
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=ch_1, kernel_size=cv_ksize, padding=cv_pad),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),

            nn.Conv3d(in_channels=ch_1, out_channels=ch_2, kernel_size=cv_ksize, padding=cv_pad),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd),

            nn.Conv3d(in_channels=ch_2, out_channels=ch_3, kernel_size=cv_ksize, padding=cv_pad),
            nn.BatchNorm3d(ch_3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=mp_ksize, stride=mp_strd)
        )
        self.fc_input_size = self._calculate_fc_input_size(maxes)
        self.fc1 = nn.Linear(self.fc_input_size, hidden)  # Calculate the size after 3D convolutions
        self.fc2 = nn.Linear(hidden, prms)
        self.fc25 = nn.Linear(prms,prms)
        self.fc3 = nn.Linear(prms,1)
        
    def _calculate_fc_input_size(self,maxes):
        # Dummy input to calculate feature map size after the convolutions
        dummy_input = torch.rand(1, 1, int(maxes[0]//self.target_resolution), 
                                 int(maxes[1]//self.target_resolution), 
                                 int(maxes[2]//self.target_resolution))
        x = self.conv_layers(dummy_input)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1,self.fc_input_size)
        x = self.activate(self.fc2(self.activate(self.fc1(x))))
        y = self.activate(self.fc25(x))
        x = x+y
        x = self.fc3(x)
        return x
class mask_position(nn.Module):
    def __init__(self, max_len=1000, dropout=0.1, div_multi=10000):
        super(mask_position, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.div_multi = div_multi
        self.max_len = max_len
    
    def generate_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        self.max_len = max_len
        div_term = torch.exp(torch.arange(0, 2*d_model, 2).float() * (-math.log(self.div_multi)))
        pe = torch.zeros(max_len, d_model)
        
        for i in range(d_model):
            if i % 2 == 0:
                pe[:, i] = torch.sin(position / div_term[i]).squeeze(1)
            else:
                pe[:, i] = torch.cos(position / div_term[i]).squeeze(1)
        return pe.unsqueeze(0).transpose(0, 1)
    
    def forward(self, x):
        max_len, d_model = x.size(1), x.size(-1)
        pe = self.generate_positional_encoding(max_len, d_model)
        pe = pe.to(x.device)
        x = x + pe.squeeze(1)
        return self.dropout(x)
    
class mask_transformer(nn.Module):
    def __init__(self, input_dims, nhead, dim_ff, output_dims, dropout = 0.1):
        super(mask_transformer,self).__init__()
        self.position = mask_position()
        
        self.self_attn = nn.MultiheadAttention(input_dims,nhead,dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dims, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff,input_dims)
        )
        self.norm1 = nn.LayerNorm(input_dims)
        self.norm2 = nn.LayerNorm(input_dims)
        self.lstm = nn.LSTM(input_size=input_dims,hidden_size=dim_ff,batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff,output_dims * 2),
            nn.ReLU(),
            nn.Linear(output_dims*2,output_dims)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.position(x)
        mask_attn, _ = self.self_attn(x,x,x)
        x=x+self.dropout(mask_attn)
        x=self.norm1(x)
        ff = self.feed_forward(x)
        x=x+self.dropout(ff)
        x=self.norm2(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)        
        x=self.dense_layers(x)
        return x
