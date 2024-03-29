# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:21:53 2023

@author: arwilzman
"""
import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import Image, ImageSequence
from sklearn.cluster import KMeans
from scipy.stats import t

date = datetime.datetime.now()
month = date.strftime("%m")
day = date.strftime("%d")
year = date.strftime("%y")
date = ('_'+month + '_' + day + '_' + year)
directory = 'Z:/_Current IRB Approved Studies/Jumping_Study/'
total = pd.read_csv(directory+f'Data/Jump_Study_Data{date}.csv')

plot_fig = True
save_fig = False
grif = False

def plot_2drelation(pred, val, figsize=None, title='', xlab='', ylab='', 
                    colors=None, ylim=None, xlim=None):
    # Create a new figure with the specified figsize or use the default size
    if figsize:
        plt.figure(figsize=figsize)
    if colors is not None:
        plt.scatter(pred, val, c=colors, cmap='cool')
        cb = plt.colorbar(orientation='horizontal',ticks=[1,2])
        cb.set_label(label='Landing Limbs',size=20)
        cb.ax.tick_params(labelsize=25)
    else:
        plt.scatter(pred, val)
    plt.title(title,fontsize=20)
    plt.ylabel(ylab,fontsize=20)
    plt.xlabel(xlab,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)  
    plt.tight_layout()
        
def pngs_to_gif(file_list, export_filename=None, duration=3000, loop=0):
    images = [Image.open(filename) for filename in file_list]
    # Assuming all images have the same size, you can use the size of the first image
    size = images[0].size
    if export_filename is None:
        export_filename = fd.asksaveasfilename(defaultextension=".gif")
        if not export_filename:
            return  # User canceled the save dialog
    with Image.new("RGB", size) as gif:
        gif.save(export_filename, save_all=True, append_images=images, duration=duration, loop=loop)

def corr_analysis(corr, cols, title, plot_fig, save_fig):
    # Initialize a DataFrame to store the results
    result_df = pd.DataFrame(index=cols, columns=cols, dtype=str)
    
    # Calculate significance level (e.g., alpha = 0.05)
    alpha = 0.05
    
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            correlation = corr[cols[i]][cols[j]]
            p_value = 2 * (1 - t.cdf(abs(correlation) * np.sqrt(len(corr) - 2), df=len(corr) - 2))
            if p_value < alpha:
                significance = "significant"
            else:
                significance = "not significant"
            
            # Store the correlation coefficient and p-value in the DataFrame
            result_df.at[cols[i], cols[j]] = f"{correlation:.4f}, p-value: {p_value:.4f}, {significance}"
            result_df.at[cols[j], cols[i]] = f"{correlation:.4f}, p-value: {p_value:.4f}, {significance}"

    # Save the DataFrame to a text file
    result_df.to_csv(f'{directory}/Data/{title}_{date}.csv', sep=',')
    
    # Plot the correlation matrix
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=range(len(cols)), labels=cols, rotation='vertical')
    plt.yticks(ticks=range(len(cols)), labels=cols)
    plt.title(title)
    
    # Save or show the plot
    if save_fig:
        plt.savefig(f'{directory}/Data/graphpics/{title}.png', bbox_inches='tight', dpi=500)
    if plot_fig:
        plt.show()

predictor_col = ['Hip Flexion ROM', 'Knee Flexion ROM',
                 'Ankle Flexion ROM', 'Hip Flexion at Contact',
                 'Knee Flexion at Contact', 'Ankle Flexion at Contact']
val_col = ['DIS','IMU_FFT',
           'RXF','RXF_R','RXF_FFT',
           'JCF', 'JCF_R','JCF_FFT',
           'RXF_FE','RXF_FE_FFT','RXF_SMR',
           'JCF_FE','JCF_FE_FFT','JCF_SMR']

#%%
num_rows = 5
num_cols = 3
figsize = [14,18]
fontsize = 15
colors = ['fuchsia','cyan']
for i, pred in enumerate(total.columns):
    counts = np.zeros(num_rows)
    if pred in val_col:
        continue
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    for j, val in enumerate(val_col):
        if val == 'DIS' or val == 'IMU_FFT':
            row = 0
        elif val == 'RXF' or val == 'RXF_R' or val == 'RXF_FFT':
            row = 1
        elif val == 'JCF' or val == 'JCF_R' or val == 'JCF_FFT':
            row = 2
        elif 'RXF' in val:
            row = 3
        else:
            row = 4
            
        ax = axs[row, int(counts[row])]
        if pred == 'Landing Limbs':
            box1 = []
            box2 = []
            box1 = total[total['Landing Limbs']<1.5][val].abs().dropna()
            box2 = total[total['Landing Limbs']>1.5][val].abs().dropna()
            box1.dropna()
            box2.dropna()
            box_data = [box1,box2]
            box = ax.boxplot(box_data,patch_artist=True,labels=[1,2])
            box['boxes'][0].set_facecolor('cyan')
            box['boxes'][1].set_facecolor('fuchsia')
            ax.tick_params(axis='x',labelsize=fontsize*1.5)
        else:    
            ax.scatter(total[pred], abs(total[val]), 
                       c=total['Landing Limbs'], cmap='cool', edgecolors='black')
        ax.set_ylabel(val,fontsize=fontsize)
        if counts[row] == 1 and row == 0:
            ax.set_title(f'{pred}',fontsize=fontsize*2)    
        counts[row] += 1
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{directory}Data/graphpics/{pred}_all_outcomes.png',dpi=500)
    plt.show()
#%%

clusters = [2, 3]
num_rows = 5
num_cols = 3
figsize = [14,18]
fontsize = 15

for clust in clusters:
    kmeans = KMeans(n_clusters=clust)
    kmeans.fit(total[predictor_col])
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # Create a new figure for each cluster
    for i, pred in enumerate(total.columns):
        counts = np.zeros(num_rows)
        if pred in val_col:
            continue
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        for j, val in enumerate(val_col):
            if val == 'DIS' or val == 'IMU_FFT':
                row = 0
            elif val == 'RXF' or val == 'RXF_R' or val == 'RXF_FFT':
                row = 1
            elif val == 'JCF' or val == 'JCF_R' or val == 'JCF_FFT':
                row = 2
            elif 'RXF' in val:
                row = 3
            else:
                row = 4
            ax = axs[row, int(counts[row])]
            ax.set_title(f'{clust}-cluster, kmeans (unsupervised✔)')
            ax.scatter(total[pred], total[val], c=labels, cmap='viridis', edgecolors='black')
            ax.set_xlabel(pred + '✔' if pred in predictor_col else pred)
            ax.set_ylabel(val)
            if counts[row] == 1 and row == 0:
                ax.set_title(f'{pred}',fontsize=fontsize*2)    
            counts[row] += 1
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{directory}Data/MLpics/{pred}_kmeans_{clust}clusters.png')
        plt.show()

#%%
if plot_fig:
    for pred in predictor_col:
        for val in val_col:
            plot_2drelation(total[pred],total[val],[7,7],f'{pred} vs. {val}',
                            pred,val,total['Landing Limbs'])
            if save_fig:
                plt.savefig(f'{directory}Data/graphpics/{pred}_{val}.png')
            plt.show()
        
#%%
step_size = 45
a_lim = [0,360]
figsize=[12,12]
fontsize = figsize[0]*2
num_pics = int((a_lim[1]-a_lim[0])/step_size)
if grif:
    if plot_fig:
        for val in val_col:
            rang = total[total[val]<10e10][val].max()-total[val].min()
            for i in range(len(predictor_col)-1):
                for j in range(len(predictor_col)-1-i):
                    gif_files = []
                    elev=5
                    azim=a_lim[0]
                    roll=0
                    flag1 = False
                    flag2 = False
                    flag3 = FalseF
                    for k in range(num_pics):
                        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=figsize)
                        surf = ax.scatter(total[predictor_col[i]],
                                          total[predictor_col[i+1+j]],
                                          total[val], c=total['Landing Limbs'],
                                          cmap='cool', edgecolors='black', linewidths=1,
                                          s=total['Mass']**2/(total['Mass'].max()-total['Mass'].min()))
                        # Add texts as axis labels
                        ax.set_xlabel(f'{predictor_col[i]}', fontsize=fontsize, color='green', labelpad=fontsize)
                        ax.set_ylabel(f'{predictor_col[i+1+j]}', fontsize=fontsize, color='green', labelpad=fontsize)
                        
                        ax.tick_params(axis='both', which='major', labelsize=fontsize)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))
                        ax.zaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))
                        if total[total[val]<10e10][val].max() < 0:
                            ax.invert_zaxis()
                            ax.text(total[predictor_col[i]].max()/2,
                                    total[predictor_col[i+j+1]].max()/2,
                                    total[val].min()-rang*.2, f'{val}', fontsize=fontsize, color='green')
                        else:
                            ax.text(total[predictor_col[i]].max()/2,
                                    total[predictor_col[i+j+1]].max()/2,
                                    total[total[val]<10e10][val].max()+rang*.2, f'{val}', fontsize=fontsize, color='green')
                        if flag2:
                            azim-=step_size
                            if azim < a_lim[0]:
                                flag2 = False
                        else:
                            azim+=step_size
                            if azim > a_lim[1]:
                                flag2 = True
                        ax.view_init(elev, azim, roll)
                        
                        plt.show()
                        if save_fig:
                            gif_file = f'{directory}Data/graphpics/{val}_{predictor_col[i]}_{predictor_col[i+1+j]}_{len(gif_files)}.png'
                            gif_files.append(gif_file)
                            fig.savefig(gif_file,bbox_inches='tight')
                    if save_fig:
                        pngs_to_gif(gif_files,export_filename=directory+f'Data/graphgifs/{val}_{predictor_col[i]}_{predictor_col[i+1+j]}.gif')

#%%
pred_corr = total[predictor_col].corr()
val_corr = total[val_col].corr()

corr_analysis(pred_corr,predictor_col,'Predictor Correlations',plot_fig,save_fig)
corr_analysis(val_corr,val_col,'Outcome Correlations',plot_fig,save_fig)