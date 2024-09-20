# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:37:03 2024

@author: Andrew
"""
import numpy as np
from pyDOE2 import lhs


direct = 'A:/Work/Code/turing_bash/'
# Define the hyperparameter ranges
H1_values = [64, 128, 256, 512]
LYR_values = [1, 2, 3]
LR_values = [1e-2, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3]
total_combinations = len(H1_values) * len(LYR_values) * len(LR_values)
# Number of experiments
num_experiments = 16

# Perform Latin Hypercube Sampling
lhs_sample = lhs(3, samples=num_experiments)  # 3 dimensions: H1, LYR, LR

# Map LHS samples to actual values
H1_indices = np.floor(lhs_sample[:, 0] * len(H1_values)).astype(int)
LYR_indices = np.floor(lhs_sample[:, 1] * len(LYR_values)).astype(int)
LR_indices = np.floor(lhs_sample[:, 2] * len(LR_values)).astype(int)

# Generate the 10 parameter sets based on the LHS sample
parameter_sets = [(H1_values[H1_indices[i]], LYR_values[LYR_indices[i]], LR_values[LR_indices[i]])
                  for i in range(num_experiments)]

# Create .sh files
for i, (H1, LYR, LR) in enumerate(parameter_sets):
    filename = f"{direct}detden{i+1}.sh"
    with open(filename, 'w', newline='\n') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n 8\n")
        f.write("#SBATCH --mem=32G\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -t 01:00:00\n")
        f.write(f"python density_training.py -a --batch 32 -h1 {H1} --layers {LYR} -lr {LR} -e 20 --name detden{i+1}\n")

print(f"{num_experiments}/{total_combinations} .sh files generated.")
