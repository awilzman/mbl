# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:29:34 2025

@author: arwilzman

Please rename the dated file to the main file name "LHS.csv"
If you want to use that version. The LHS.csv is searched when using
readodb.py
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from datetime import datetime

# Get current date
current_date = datetime.now()
formatted_date = current_date.strftime("%m_%d_%Y")

    # Tsai-Wu failure criteria parameters
    # https://www.sciencedirect.com/science/article/pii/S1751616111001445
    # https://www.sciencedirect.com/science/article/pii/S8756328214003664
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC4996317/#b127
    # https://www.sciencedirect.com/science/article/pii/S002192900800599X
    # https://www.tandfonline.com/doi/epdf/10.3109/17453678108992136?needAccess=true
    # https://www.sciencedirect.com/science/article/pii/S1751616113000787
    # https://www.sciencedirect.com/science/article/pii/S8756328214003664
    

# Define the number of experiments and parameter limits
#this needs to match everywhere
num_exp = 64  # Number of experiments
axial_limits = [-0.015, -0.005, 0.005, 0.0002]  # strain mm/mm
tsv_limits = [-0.015, -0.005, 0.005, 0.0002]
s_limits = [0.0005, 0.01]  # Ensure min/max order

# Create LHS sampler
sampler = qmc.LatinHypercube(d=5)  # 5 parameters: X_c, Y_c, X_t, Y_t, S
samples = sampler.random(n=num_exp)

# Scale the samples to match parameter ranges
X_c_samples = qmc.scale(samples[:, 0].reshape(-1, 1), axial_limits[0], axial_limits[1]).flatten()
Y_c_samples = qmc.scale(samples[:, 1].reshape(-1, 1), tsv_limits[0], tsv_limits[1]).flatten()
X_t_samples = qmc.scale(samples[:, 2].reshape(-1, 1), axial_limits[3], axial_limits[2]).flatten()
Y_t_samples = qmc.scale(samples[:, 3].reshape(-1, 1), tsv_limits[3], tsv_limits[2]).flatten()
S_samples = qmc.scale(samples[:, 4].reshape(-1, 1), s_limits[0], s_limits[1]).flatten()

# Create a DataFrame to store the scaled samples
data = {
    'X_c': X_c_samples,
    'Y_c': Y_c_samples,
    'X_t': X_t_samples,
    'Y_t': Y_t_samples,
    'S': S_samples
}

df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
csv_file = f'Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/Cadaver_Data/LHS_{formatted_date}.csv'
df.to_csv(csv_file, index=False)

print(f'LHS samples (scaled) have been saved to {csv_file}')
