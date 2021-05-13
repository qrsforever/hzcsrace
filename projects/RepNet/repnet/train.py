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
FRAME_WIDTH, FRAME_HEIGHT = 112, 112
NUM_FRAMES = 64
NUM_DMODEL = 512
REP_OUT_TIME_RATE = 0.12

num_epochs = 30
lr = 0.001
device = torch.device("cuda")

def train(opt):

    train_dataset = CountixDataset(DATASET_PREFIX, 'train')
    model = RepNet(NUM_FRAMES, NUM_DMODEL, device)

    ckpt_path = f'{DATASET_PREFIX}/repnet.pt'

    # if os.path.exists(ckpt_path):
        # checkpoint = torch.load(ckpt_path)
        # model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)

    if opt.local_rank == -1:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4)
        model.cuda()
    else:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, sampler=sampler)
        model = DDP(model.cuda(), device_ids=[opt.local_rank], output_device=opt.local_rank)

    optimizer = O.Adam(model.parameters(), lr=lr)
    # scheduler = O.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.92)
    scheduler = O.lr_scheduler.MultiStepLR(optimizer, milestones=[
        3, 10, 20, 40, 70, 100, 200, 250, 300, 350, 400], gamma=0.6)
    # scheduler = O.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
    loss_mae = nn.SmoothL1Loss()
    loss_mse = nn.MSELoss()
    loss_bce = nn.BCEWithLogitsLoss()

    torch.cuda.empty_cache()
    fw = open(f'{DATASET_PREFIX}/metrics.json', 'w')
    for epoch in tqdm(range(0, opt.num_epochs)):
        if opt.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        period_length_loss = []
        period_predict_loss = []
        period_count_loss = []
        train_loss = []
        curr_lr = scheduler.get_last_lr()
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (X, y1, y2, y3) in enumerate(pbar):
            X = X.to(device)
            y1, y2, y3 = y1.to(device), y2.to(device), y3.to(device)
            
            y1_pred, y2_pred = model(X)
            # y3_pred = torch.sum((y2_pred > 0) / (y1_pred + 1e-1), 1)
            
            loss1 = loss_mae(y1_pred, y1)
            loss2 = loss_bce(y2_pred, y2)
            # loss3 = loss_mae(y3_pred, y3)
            loss = loss1 + 8*loss2 # + loss3
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            # period_length_loss.append(loss1.item())
            # period_predict_loss.append(loss2.item())
            # period_count_loss.append(loss3.item())
            train_loss.append(loss.item())

            metrics = {
		'Epoch': epoch,
		'LR': '%.6f' % curr_lr[0],
		'Loss1': '%.3f' % loss1.item(),
		'Loss2': '%.3f' % loss2.item(),
		# 'Loss3': '%.3f' % loss3.item(),
		'Loss': '%.3f' % np.mean(train_loss)
	    }
            pbar.set_postfix(metrics)
            fw.write(json.dumps(metrics))
            fw.write('\n')
            fw.flush()

        scheduler.step()
        
        # TODO
        # for name, params in model.named_parameters():
        #     if name == 'trans1.trans_encoder.1.layers.0.self_attn.out_proj.weight':
        #         print(params.data.detach().cpu().numpy()[0, :6])

        checkpoint = {
            'model_state_dict': model.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
    fw.close()


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

    train(args)
