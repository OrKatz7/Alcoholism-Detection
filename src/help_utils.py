import torch
import os
import random
import numpy as np
import time
import math
from logging import getLogger, INFO, FileHandler,  Formatter, StreamHandler
from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold,GroupKFold


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def split_kfold(train,drop_c = None,n_splits=5,seed=42,groups='id',verbose=True,CFG=None):
    folds = train.copy()
    folds = folds[folds.stimuli=='S2 match']
    for d in drop_c:
        folds = folds[folds.stimuli != d]
    folds = folds.reset_index(drop=True)
    Fold = GroupKFold(n_splits=CFG.n_fold)#StratifiedKFold(n_splits=CFG.n_fold)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col],folds['id'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    folds = folds.drop('Unnamed: 0',1)
    if verbose:
        print(folds.groupby(['fold', CFG.target_col]).size())
    return folds

def on_hot(x,s=3):
    y = np.zeros([s])
    y[x]=1
    return y
