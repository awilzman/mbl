#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 08:00:00
python __init__.py -n arw -a -d --name arw_10k --batch 64 --epochs 10000 -lr 1e-4 --hidden1 512 --hidden2 256 --hidden3 169 --noise 5