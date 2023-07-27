#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -mem 4G
#SBATCH -gpu 1
#SBATCH --partition short

python main.py --bone 'tibia' --model 'M2Q' --save 'first'