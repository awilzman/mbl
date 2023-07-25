# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# sz needs to be defined as data shape[1]
# networks:
#   informed_net:
#
#   FC_net:
# 
#   mask2qct_net:
# 
#   HR_mask2qct_net:
# 
#   netG:
# 
#   netD:
# =============================================================================
import pandas as pd
import numpy as np
import torch
from torch import nn


class informed_net(nn.Module):
    def __init__(self,epi_ind,met_ind,dia_ind,tot_ind,trunc,prms):
        super(informed_net, self).__init__()
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
    def __init__(self,layer1_size,layer2_size,sz):
        super(FC_net, self).__init__()
        
        self.lay1 = nn.Linear(sz,layer1_size)
        self.lay2 = nn.Linear(layer1_size,layer2_size)
        self.lay3 = nn.Linear(layer2_size,layer1_size)
        self.lay4 = nn.Linear(layer1_size,sz)
        
        self.activate = nn.LeakyReLU()
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

class mask2qct_net(nn.Module):
    def __init__(self,prms,maxes,target_resolution,hidden=64,mp_ksize=2,mp_strd=2,
                 cv_ksize=2,cv_pad=1,ch_1=4,ch_2=16,ch_3=32,dropout=0.2):
        super(mask2qct_net, self).__init__()
        
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
        dummy_input = torch.rand(1, 1, int(maxes[0]//target_resolution), 
                                 int(maxes[1]//target_resolution), int(maxes[2]//target_resolution))
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
        super(HR_mask2qct_net, self).__init__()
        self.high_res_mag = high_res_mag
        self.activate = nn.LeakyReLU()
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
        super(netG, self).__init__()
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
        super(netD, self).__init__()
        
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
