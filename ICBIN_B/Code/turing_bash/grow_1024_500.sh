#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 10:00:00
python __init__.py -n grow -d --name grow_8h_500 --hidden1 1024 --noise 4 --batch 64 --epochs 0 --traintime 25200 -lr 1e-5 --decay 5e-7 --eval_bs 40 --pint 200 --loadgen ae_grow_16h_1024_90_40_0 --numpoints 500