# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:39:07 2024
Not that kind of stock

@author: arwilzman
"""

import torch
from torch_cluster import knn_graph
import pandas as pd
import numpy as np
import h5py
import os
import open3d as o3d

k = 16

directory = '../Stock_Datasets/ModelNet10/'
file = 'train0.h5'
newfile = 'train0_knn.h5'

with h5py.File(directory+file, 'r') as hf:
    data = np.array(hf['data'])
a = torch.FloatTensor(data)

dataset_knn_list = []

# Iterate over the first dimension of the dataset
for i in range(a.size(0)):
    # Apply knn_graph function to each item in the dataset
    dataset_knn_i = knn_graph(a[i,:,:], k)
    # Append the result to the list
    dataset_knn_list.append(dataset_knn_i)

# Stack the results along a new dimension
dataset_knn = torch.stack(dataset_knn_list, dim=0)

with h5py.File(directory+newfile, 'w') as hf:
    hf.create_dataset('data_knn', data=dataset_knn)