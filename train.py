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
    args = parser.parse_args()
    CFG = eval(args.config)
    seed_torch(CFG.seed)
    OUTPUT_DIR = f"{CFG.DATA.OUTPUT_DIR}/{CFG.exp_name}/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    LOGGER = init_logger(log_file=OUTPUT_DIR+f'train_{CFG.exp_name}.log')
    train_df = pd.read_csv(CFG.DATA.img_csv_path)
    lstm_df = pd.read_parquet(CFG.DATA.sax_csv_path)
    folds = eval(CFG.k_fold_fun)(train_df,drop_c = CFG.DATA.drop_stimuli,n_splits=CFG.n_fold,seed=CFG.seed,
                                 groups=CFG.DATA.split_by,verbose=CFG.verbose,CFG=CFG)
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            end = train_loop(folds, fold , lstm_df, CFG, LOGGER)
            LOGGER.info(f"========== fold: {fold} end ==========")

    
    