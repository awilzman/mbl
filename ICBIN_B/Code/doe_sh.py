# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:37:03 2024

@author: Andrew
"""
import numpy as np
from pyDOE2 import lhs


direct = 'A:/Work/Code/turing_bash/'
# Define the hyperparameter ranges
H1_values = [2, 4, 8, 16]
LYR_values = [1, 2, 4, 6]
exp_values = [1, 2, 4]
LR_values = [5e-3, 3e-3, 1e-3]
total_combinations = len(H1_values) * len(LYR_values) * len(LR_values) * len(exp_values)
# Number of experiments
num_experiments = 8
batch = 64
epch = 50

# Perform Latin Hypercube Sampling
lhs_sample = lhs(4, samples=num_experiments)  # 4 dimensions: H1, LYR, LR, EXP

# Map LHS samples to actual values
H1_indices = np.floor(lhs_sample[:, 0] * len(H1_values)).astype(int)
LYR_indices = np.floor(lhs_sample[:, 1] * len(LYR_values)).astype(int)
LR_indices = np.floor(lhs_sample[:, 2] * len(LR_values)).astype(int)
exp_indices = np.floor(lhs_sample[:, 3] * len(exp_values)).astype(int)

# Generate the 10 parameter sets based on the LHS sample
parameter_sets = [(H1_values[H1_indices[i]], LYR_values[LYR_indices[i]], 
                   LR_values[LR_indices[i]], exp_values[exp_indices[i]])
                  for i in range(num_experiments)]

# Create .sh files
for i, (H1, LYR, LR, exp) in enumerate(parameter_sets):
    filename = f"{direct}lstm{i+2}.sh"
    
    line = f"python density_training.py -a --batch {batch} -h1 {H1} "
    line += f"--experts {exp} --layers {LYR} -lr {LR} -e {epch} --name lstm_adam{i+2}\n"
    with open(filename, 'w', newline='\n') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n 8\n")
        f.write("#SBATCH --mem=32G\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -t 08:00:00\n")
        f.write(line)

print(f"{num_experiments}/{total_combinations} .sh files generated.")
