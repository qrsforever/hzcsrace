#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file train.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-12 14:32


import os, math, json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import numpy as np
from torchvision import models as M
from torchvision import transforms as T

from repnet.data.countix.dataset import CountixDataset
from repnet.models.repnet import RepNet

from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader


DATASET_PREFIX = '/data/datasets/cv/countix'
NUM_FRAMES = 64
NUM_DMODEL = 512
CKPT_PATH = f'{DATASET_PREFIX}/last.pt'


def train(device, model, pbar, optimizer, criterions, metrics_callback=None):
    model.train()
    loss_list = []
    for X, y1, y2, y3 in pbar:
        X = X.to(device)
        y1, y2, y3 = y1.to(device), y2.to(device), y3.to(device)

        y1_pred, y2_pred = model(X)

        loss1 = criterions[0](y1_pred, y1)
        loss2 = criterions[1](y2_pred, y2)

        loss = loss1 + 3*loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if metrics_callback is not None:
            metrics_callback(
                '%.6f' % np.mean(loss_list),
                '%.6f' % loss1.item(),
                '%.6f' % loss2.item())

        del X, y1, y2, y3, y1_pred, y2_pred
    return np.mean(loss_list)


def valid(device, model, pbar, criterions, metrics_callback=None):
    model.eval()
    loss_list = []
    for X, y1, y2, y3 in pbar:
        X = X.to(device)
        y1, y2, y3 = y1.to(device), y2.to(device), y3.to(device)

        y1_pred, y2_pred = model(X)

        loss1 = criterions[0](y1_pred, y1)
        loss2 = criterions[1](y2_pred, y2)
        loss = loss1 + 3*loss2
        loss_list.append(loss.item())

        if metrics_callback is not None:
            metrics_callback(
                '%.6f' % np.mean(loss_list),
                '%.6f' % loss1.item(),
                '%.6f' % loss2.item())

        del X, y1, y2, y3, y1_pred, y2_pred
    return np.mean(loss_list)


def inference(device, model, pbar, metrics_callback=None):
    # TODO only one test
    for X, y1, y2, y3_true in pbar:
        X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
        y1_pred, y2_pred = model(X)
        
        y3_pred = torch.round(torch.sum((y2_pred > 0) / (y1_pred + 1e-1), 1))
        y3_calc = torch.round(torch.sum((y2 > 0) / (y1 + 1e-1), 1))
        
        if metrics_callback is not None:
            metrics_callback(
                y3_pred.cpu().numpy().flatten().astype(int).tolist(),
                y3_calc.cpu().numpy().flatten().astype(int).tolist(),
                y3_true.numpy().flatten().astype(int).tolist())

        break


def train_loop(opt, model, ckpt_path,
               train_loader, valid_loader, test_loader,
               optimizer, scheduler, criterions, device):

    metrics_writer = open(f'{DATASET_PREFIX}/metrics.json', 'w')
    start_epoch = 0

    # load model
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        if opt.local_rank != -1:
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    lr = optimizer.param_groups[0]['lr']

    for epoch in tqdm(range(start_epoch, opt.num_epochs + start_epoch)):
        # train
        train_loader.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        with tqdm(train_loader, total=len(train_loader), desc='train') as pbar:
            train_loss = train(device, model, pbar, optimizer, criterions,
                         lambda loss, loss1, loss2: pbar.set_postfix(
                             epoch=epoch, lr=lr, loss=loss, loss1=loss1, loss2=loss2))

            metrics_writer.write('Train: {}\n'.format(pbar))

        # valid
        valid_loader.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        with torch.no_grad():
            with tqdm(valid_loader, desc='valid') as pbar:
                valid_loss = valid(device, model, pbar, criterions,
                             lambda loss, loss1, loss2: pbar.set_postfix(
                                 epoch=epoch, lr=lr, loss=loss, loss1=loss1, loss2=loss2))
                metrics_writer.write('Valid: {}\n'.format(pbar))

        # inference only one
        test_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            with tqdm(test_loader, desc='Inference') as pbar:
                inference(device, model, pbar,
                          lambda y_pred, y_calc, y_true: pbar.set_postfix(
                              y_pred=y_pred, y_calc=y_calc, y_true=y_true))
                metrics_writer.write('Inference: {}\n'.format(pbar))

        # update learning rate
        if isinstance(scheduler, O.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss) # todo
            lr = '%.8f' % scheduler._last_lr[0]
        else:
            scheduler.step()
            lr = '%.8f' % scheduler.get_last_lr()[0]

        metrics_writer.flush()

        # save model
        if ckpt_path is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if opt.local_rank == -1 else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }
            torch.save(checkpoint, ckpt_path)

    metrics_writer.close()


def run_train(opt):

    device = torch.device("cuda")
    model = RepNet(NUM_FRAMES, NUM_DMODEL).to(device)

    # data loader
    train_dataset = CountixDataset(DATASET_PREFIX, 'train')
    valid_dataset = CountixDataset(DATASET_PREFIX, 'val')
    test_dataset = CountixDataset(DATASET_PREFIX, 'test')

    if opt.local_rank == -1:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=False)
    else:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                num_workers=4, drop_last=True, sampler=train_sampler)

        valid_sampler = DistributedSampler(valid_dataset, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers=4, sampler=valid_sampler)

        test_sampler = DistributedSampler(test_dataset, shuffle=True)
        test_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers=4, sampler=test_sampler)

        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                find_unused_parameters=True)

    # hyper parameters
    optimizer = O.Adam(model.parameters(), lr=0.0001)
    # optimizer = O.SGD(model.parameters(), lr=lr)
    # scheduler = O.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # scheduler = O.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
    # scheduler = O.lr_scheduler.MultiStepLR(optimizer, milestones=[
    #         3, 10, 50, 100, 200, 300, 400], gamma=0.6)
    scheduler = O.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6, min_lr=1e-8)
    loss_mae = nn.SmoothL1Loss()
    loss_bce = nn.BCEWithLogitsLoss()
    criterions = [loss_mae, loss_bce]

    train_loop(opt, model, CKPT_PATH, \
               train_loader, valid_loader, test_loader, \
               optimizer, scheduler, criterions, device)

    if opt.local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--local_rank',
            default=-1,
            type=int,
            dest='local_rank',
            help="")
    parser.add_argument(
            '--num_epochs',
            default=30,
            type=int,
            dest='num_epochs',
            help="")
    parser.add_argument(
            '--batch_size',
            default=10,
            type=int,
            dest='batch_size',
            help="")

    args = parser.parse_args()

    run_train(args)
