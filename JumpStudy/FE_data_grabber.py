import os
import numpy as np
import pandas as pd

# I think this code is obsolete because dataloader just finds the file itself
directory = 'Z:/_Current IRB Approved Studies/Jumping_Study/'
std_fe = '/FE/FE_STD.TXT' #standard FE file name
crtx_fe = '/FE/FE_CRTX.TXT' #crtx FE file name
# data columns to grab
data = pd.DataFrame(columns=['SampName','type','S','F.ult','(Tb.F/TF)prox','(Tb.F/TF)dist','E.app','Tb.VM','C.VM','TB.ES','C.ES'])
#right now I only use E.app but the others can be useful

#column lists for upcoming dataframes
collist = ['L_ANK_MAX_FORCE_Z_Distance_Jump','R_ANK_MAX_FORCE_Z_Distance_Jump',
           'L_ANK_MAX_FORCE_Z_Drop_Step','R_ANK_MAX_FORCE_Z_Drop_Step',
           'L_ANK_MAX_FORCE_Z_Drop_Low','R_ANK_MAX_FORCE_Z_Drop_Low',
           'L_ANK_MAX_FORCE_Z_Drop_Med','R_ANK_MAX_FORCE_Z_Drop_Med',
           'L_ANK_MAX_FORCE_Z_Drop_High','R_ANK_MAX_FORCE_Z_Drop_High',
           'L_ANK_MAX_FORCE_Z_L_SL_Drop_Step','R_ANK_MAX_FORCE_Z_R_SL_Drop_Step',
           'L_ANK_MAX_FORCE_Z_L_SL_Drop_Low','R_ANK_MAX_FORCE_Z_R_SL_Drop_Low',
           'L_ANK_MAX_FORCE_Z_L_SL_Drop_Med','R_ANK_MAX_FORCE_Z_R_SL_Drop_Med',
           'L_ANK_MAX_FORCE_Z_L_SL_Drop_High','R_ANK_MAX_FORCE_Z_R_SL_Drop_High']
fcollist = ['Participant']+collist

#grab FE data
for i in os.listdir():
    if i.startswith('JS_'):
        if os.path.isfile(i+std_fe):
            A = pd.read_csv(i+std_fe,sep='\t')
            A['type'] = 'std'
            data = pd.concat([data, A[['SampName','type','S','F.ult','(Tb.F/TF)prox','(Tb.F/TF)dist','E.app','Tb.VM','C.VM','TB.ES','C.ES']]])
        if os.path.isfile(i+crtx_fe):
            B = pd.read_csv(i+crtx_fe,sep='\t')
            B['type'] = 'crtx'
            data = pd.concat([data, B[['SampName','type','S','F.ult','(Tb.F/TF)prox','(Tb.F/TF)dist','E.app','Tb.VM','C.VM','TB.ES','C.ES']]])
            
#grab forces from jump study data            
forces = pd.read_excel('Data/JS_Data.xlsx')
forces = forces[fcollist]
strains = np.zeros((2,len(forces),len(forces.columns)-1))

#calculate max forces / relevant E.app
#note that E.app is calculated only from right tibia, so R/L comparisons will not be valid
k=0
pts = ['JS_1','JS_2','JS_S3','JS_S4','JS_S5','JS_6','JS_7','JS_S8','JS_S9','JS_S10','JS_F11','JS-F12']
std,crtx = np.zeros(len(pts)),np.zeros(len(pts))
for col in forces.drop(columns=['Participant']):
    j=0
    for i in pts:
        ram = np.array(data.loc[data['SampName']==i]['S'])
        if ram.shape[0] == 2:
            std[j] = forces.loc[forces['Participant']==j+1][col][j]/np.array(data.loc[data['SampName']==i]['S'])[0]
            crtx[j] = forces.loc[forces['Participant']==j+1][col][j]/np.array(data.loc[data['SampName']==i]['S'])[1]
        elif ram.shape[0] == 1:
            if data.loc[data['SampName']==i]['type'][0] == 'std':
                std[j] = forces.loc[forces['Participant']==j+1][col][j]/np.array(data.loc[data['SampName']==i]['S'])[0]
            elif data.loc[data['SampName']==i]['type'][0] == 'crtx':
                crtx[j] = forces.loc[forces['Participant']==j+1][col][j]/np.array(data.loc[data['SampName']==i]['S'])[0]
        j+=1
    strains[0,:,k] = std
    strains[1,:,k] = crtx
    k+=1
k=0

#%%save
pdstd = pd.DataFrame(strains[0,:,:],columns=collist)
pdcrtx = pd.DataFrame(strains[1,:,:],columns=collist)
writer = pd.ExcelWriter(f'{directory}Data/JS_deformation.xlsx',engine='xlsxwriter')
pdstd.to_excel(writer,sheet_name='std')
pdcrtx.to_excel(writer,sheet_name='crtx')
writer.save()