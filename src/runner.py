import numpy as np
import torch
from data import TrainDataset,get_transforms
from torch.utils.data import DataLoader, Dataset
import models
import time
from help_utils import *
import sklearn
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=0.1, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

def get_id(row):
    return row.split("/")[-1].split(".")[0]

def train_fn(train_loader,train_loader2, model, criterion, optimizer, epoch, scheduler, device, CFG):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (batch1, batch2) in enumerate(zip(train_loader,train_loader2)):
        if np.random.random()<0.0:
            images1,x_lstm,meta, labels1,file_name = batch1
            images2,x_lstm,meta, labels2,file_name = batch2

            labels1 = labels1.to(device, non_blocking=True)
            labels2 = labels2.to(device, non_blocking=True)
            batch_size = labels1.size(0)

            alpha = 1.0
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            mixed_images = lam * images1 + (1 - lam) * images2
            mixed_images = mixed_images.to(device, non_blocking=True)

            new_label = torch.clip(labels1 + labels2, 0, 1)
            images, labels = mixed_images,new_label
        else:
            images,x_lstm,meta, labels,file_name = batch1
            labels = labels.to(device, non_blocking=True)
        
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        x_lstm = x_lstm.to(device).float()
        meta = meta.to(device).float()
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images,x_lstm,meta)
        loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.TORCH.gradient_accumulation_steps > 1:
            loss = loss / CFG.TORCH.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.TORCH.max_grad_norm)
        if (step + 1) % CFG.TORCH.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device,CFG):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    gt = []
    file_names = []
    start = end = time.time()
    for step, (images,x_lstm,meta, labels,file_name) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        x_lstm = x_lstm.to(device).float()
        meta = meta.to(device).float()
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images,x_lstm,meta)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        gt.append(labels.to('cpu').numpy())
        file_names.append(file_name)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    gt = np.concatenate(gt)
    file_names = np.concatenate(file_names).reshape(-1)
    return losses.avg, predictions,gt,file_names

from sklearn.metrics import confusion_matrix
def train_loop(folds, fold , lstm_df, CFG, LOGGER):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    
    train_dataset = TrainDataset(df = train_folds,lstm_df = lstm_df,config=CFG, transform=get_transforms(data='train',CFG=CFG))
    valid_dataset = TrainDataset(df = valid_folds,lstm_df = lstm_df,config=CFG, transform=get_transforms(data='valid',CFG=CFG))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.DATA.batch_size, 
                              shuffle=True, 
                              num_workers=CFG.DATA.num_workers, pin_memory=True, drop_last=True)
    
    train_loader2 = DataLoader(train_dataset, 
                              batch_size=CFG.DATA.batch_size, 
                              shuffle=True, 
                              num_workers=CFG.DATA.num_workers, pin_memory=True, drop_last=True)
        
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.DATA.batch_size, 
                              shuffle=False, 
                              num_workers=CFG.DATA.num_workers, pin_memory=True, drop_last=False)
    
    model = eval(CFG.model_fn)(CFG,**CFG.model_args)
    model.to(device)
    optimizer = eval(CFG.TORCH.optimizer)(model.parameters(),**CFG.TORCH.optimizer_args)
    scheduler = eval(CFG.TORCH.scheduler)(optimizer,**CFG.TORCH.scheduler_args)
    criterion = eval(CFG.TORCH.loss_fn)(**CFG.TORCH.loss_args)
#     criterion = FocalLoss()
    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        start_time = time.time()
        avg_loss = train_fn(train_loader,train_loader2, model, criterion, optimizer, epoch, scheduler, device,CFG)
        avg_val_loss, preds,gt,file_names = valid_fn(valid_loader, model, criterion, device,CFG)
        valid_labels = valid_folds[CFG.target_col].values
        scheduler.step()
        score = eval(CFG.score_metric)(valid_labels, preds.argmax(1))
        val_dff = pd.DataFrame()
        val_dff['path']=valid_folds['path'].values
        val_dff['proba'] = preds[:,1]
        val_dff['pred'] = preds.argmax(1)
        val_dff['gt'] = valid_labels
        val_dff['id'] = val_dff['path'].apply(get_id)
        new_val = val_dff.groupby("id").mean()
        new_pred = new_val['proba'].values > 0.5
        new_gt = new_val['gt']
        score2 = eval(CFG.score_metric)(new_gt.astype(int), new_pred.astype(int))
        

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score}')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy_id: {score2}')
        LOGGER.info(f'Epoch {epoch+1} - cm: {confusion_matrix(valid_labels, preds.argmax(1))}')
        LOGGER.info(f'Epoch {epoch+1} - cm_id: {confusion_matrix(new_gt.astype(int), new_pred.astype(int))}')


        if score > best_score:
            best_score = score

            val_dff.to_csv(CFG.DATA.OUTPUT_DIR+f'/{CFG.exp_name}_fold{fold}_best.csv',index=False)
            print("save csv")
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')

#             torch.save({'model': model.state_dict(), 
#                         'preds': preds},
#                         CFG.DATA.OUTPUT_DIR+f'{CFG.exp_name}_fold{fold}_best.pth')
    
#     check_point = torch.load(CFG.DATA.OUTPUT_DIR+f'{CFG.exp_name}_fold{fold}_best.pth')
    return True

