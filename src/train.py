import torch
import os
import pandas as pd
import numpy as np
import models
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import sklearn
from tqdm.auto import tqdm
from functools import partial
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations
import timm
import warnings 
warnings.filterwarnings('ignore')
from data import TrainDataset,get_transforms
from help_utils import timer,init_logger,seed_torch,split_kfold
import config
import argparse
import help_utils
from runner import train_loop
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='egg project - Or Katz Ofir Arbili and Dan Persil')
    parser.add_argument('--config', help = 'class from config.py')
    parser.add_argument('--sax_csv_path',default= '/sise/liorrk-group/OrDanOfir/eeg/data/dataset_SAX.parquet')
    parser.add_argument('--img_csv_path',default= '/sise/liorrk-group/OrDanOfir/eeg/data/img_train.csv')
    parser.add_argument('--exp_name',default='ensemble')
    parser.add_argument('--train_lstm',action='store_true')
    parser.add_argument('--train_cnn',action='store_true')
    parser.add_argument('--train_tabular',action='store_true')
    parser.add_argument('--fast',action='store_true')
    parser.add_argument('--ica',action='store_true')
    parser.add_argument('--sax',action='store_true')
    
    args = parser.parse_args()
    CFG = eval(args.config)

    CFG.train_lstm = args.train_lstm
    CFG.train_cnn = args.train_cnn
    CFG.train_tabular = args.train_tabular
    CFG.DATA.sax_csv_path = args.sax_csv_path
    CFG.DATA.img_csv_path = args.img_csv_path
    CFG.exp_name = args.exp_name
    CFG.LSTM.use_sax = args.sax
    if args.fast:
        CFG.trn_fold=[0]
    if args.ica:
        CFG.CNN.type_image = "ica"
    else:
        CFG.CNN.type_image = "img"
    seed_torch(CFG.seed)
    OUTPUT_DIR = f"{CFG.DATA.OUTPUT_DIR}/{CFG.exp_name}/"
    print(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    LOGGER = init_logger(log_file=OUTPUT_DIR+f'train_{CFG.exp_name}.log')
    LOGGER.info(CFG.__dict__)
    LOGGER.info(CFG.LSTM.__dict__)
    LOGGER.info(CFG.CNN.__dict__)
    LOGGER.info(CFG.DATA.__dict__)
    LOGGER.info(CFG.TABULAR.__dict__)
    LOGGER.info(CFG.TORCH.__dict__)
    train_df = pd.read_csv(CFG.DATA.img_csv_path)
    lstm_df = pd.read_parquet(CFG.DATA.sax_csv_path)
    folds = eval(CFG.k_fold_fun)(train_df,drop_c = CFG.DATA.drop_stimuli,n_splits=CFG.n_fold,seed=CFG.seed,
                                 groups=CFG.DATA.split_by,verbose=CFG.verbose,CFG=CFG)
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            end = train_loop(folds, fold , lstm_df, CFG, LOGGER)
            LOGGER.info(f"========== fold: {fold} end ==========")

    
    
