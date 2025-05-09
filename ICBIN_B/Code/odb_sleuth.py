# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:02:19 2025

@author: arwilzman
"""
import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ABAQUS model generator')
    parser.add_argument('--directory', type=str, default='Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/')
    parser.add_argument('--save_dir', type=str, default='Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/inps/Labeled/')
    args = parser.parse_args()

    sub_dir = os.path.join(args.directory, 'Subject Data/')
    item_dir = os.path.join(args.directory, 'Cadaver_Data/')
    save_dir = args.save_dir
    
    train_path = os.path.join(save_dir, 'tw_train.h5')
    test_path = os.path.join(save_dir, 'tw_test.h5')