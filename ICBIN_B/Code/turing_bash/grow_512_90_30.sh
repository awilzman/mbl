#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 10:00:00
python __init__.py -n grow -a --name grow_8h --hidden1 512 --grow_thresh 0.9 --grow_width 0.3 --batch 64 --epochs 0 --traintime 25200 --chpt 8 -lr 1e-4 --decay 1e-6 --eval_bs 40 --pint 300