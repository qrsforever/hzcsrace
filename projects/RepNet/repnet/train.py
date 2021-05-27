#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file train.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-12 14:32


import os
import re
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as O  # noqa
import shutil

from repnet.data.countix.dataset import CountixDataset
from repnet.models.repnet import RepNet

from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


DATASET_PREFIX = '/data/datasets/cv/countix'
NUM_FRAMES = 64
NUM_DMODEL = 512
TENSORBOARD = True


def train(device, model, pbar, optimizer, criterions, metrics_callback=None):
    model.train()
    loss_list = []
    for X, y1, y2, _ in pbar:
        X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
        y1_pred, y2_pred = model(X)

        loss1 = criterions[0](y1_pred, y1)
        loss2 = criterions[1](y2_pred, y2)

        # count error
        y3_pred = torch.sum((y2_pred > 0) / (y1_pred + 1e-1), 1)
        y3_calc = torch.sum((y2 > 0) / (y1 + 1e-1), 1)
        loss3 = criterions[0](y3_pred, y3_calc)

        loss = 3*loss1 + 10*loss2 + 2*loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        if metrics_callback is not None:
            metrics_callback(
                '%.3f' % np.mean(loss_list),
                '%.3f' % loss1.item(),
                '%.3f' % loss2.item(),
                '%.3f' % loss3.item())

        del X, y1, y2, y1_pred, y2_pred
    return np.mean(loss_list)


def valid(device, model, pbar, criterions, metrics_callback=None):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for X, y1, y2, _ in pbar:
            X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
            y1_pred, y2_pred = model(X)

            loss1 = criterions[0](y1_pred, y1)
            loss2 = criterions[1](y2_pred, y2)

            y3_pred = torch.sum((y2_pred > 0) / (y1_pred + 1e-1), 1)
            y3_calc = torch.sum((y2 > 0) / (y1 + 1e-1), 1)
            loss3 = criterions[0](y3_pred, y3_calc)

            loss = 3*loss1 + 10*loss2 + 2*loss3

            loss_list.append(loss.item())

            if metrics_callback is not None:
                metrics_callback(
                    '%.3f' % np.mean(loss_list),
                    '%.3f' % loss1.item(),
                    '%.3f' % loss2.item(),
                    '%.3f' % loss3.item())

            del X, y1, y2, y1_pred, y2_pred
    return np.mean(loss_list)


def inference(device, model, pbar, metrics_callback=None):
    # TODO only one test
    model.eval()
    with torch.no_grad():
        for X, y1, y2, y3_true in pbar:
            X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
            y1_pred, y2_pred = model(X)

            y3_pred = torch.round(torch.sum((y2_pred > 0) / (y1_pred + 1e-1), 1))
            y3_calc = torch.round(torch.sum((y2 > 0) / (y1 + 1e-1), 1))

            if metrics_callback is not None:
                metrics_callback(
                        y3_pred.cpu().numpy().flatten().astype(int).tolist()[:8],
                        y3_calc.cpu().numpy().flatten().astype(int).tolist()[:8],
                        y3_true.numpy().flatten().astype(int).tolist()[:8])

            break


def train_loop(opt, model,
               train_loader, valid_loader, test_loader,
               optimizer, scheduler, criterions, device):

    start_epoch = 0
    fmode = 'w+'

    if TENSORBOARD:
        log_dir = f'{opt.data_root}/output'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)
        tb = SummaryWriter(log_dir=log_dir, comment='repnet')

    # load model
    if os.path.exists(opt.ckpt_from_path):
        checkpoint = torch.load(opt.ckpt_from_path)
        if opt.local_rank != -1:
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint['epoch'] + 1
        fmode = 'a+'

    # metrics log
    metrics_writer = open(f'{opt.data_root}/metrics.txt', fmode)

    lr = optimizer.param_groups[0]['lr']

    for epoch in tqdm(range(start_epoch, opt.num_epochs + start_epoch), ascii=True):
        # train
        if opt.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
            valid_loader.sampler.set_epoch(epoch)

        torch.cuda.empty_cache()
        with tqdm(train_loader, total=len(train_loader), desc='train', ascii=True) as pbar:
            train_loss = train(device, model, pbar, optimizer, criterions,
                         lambda loss, loss1, loss2, loss3: pbar.set_postfix(
                             epoch=epoch, lr=lr, loss=loss, loss1=loss1, loss2=loss2, loss3=loss3))

            metrics_writer.write('{}\n'.format(pbar))

        # valid
        torch.cuda.empty_cache()
        with tqdm(valid_loader, desc='valid', ascii=True) as pbar:
            valid_loss = valid(device, model, pbar, criterions,
                         lambda loss, loss1, loss2, loss3: pbar.set_postfix(
                             epoch=epoch, lr=lr, loss=loss, loss1=loss1, loss2=loss2, loss3=loss3))
            metrics_writer.write('{}\n'.format(pbar))

        # inference
        torch.cuda.empty_cache()
        with tqdm(test_loader, desc='inference test', ascii=True) as pbar:
            inference(device, model, pbar,
                      lambda y_pred, y_calc, y_true: pbar.set_postfix(
                          y_pred=y_pred, y_calc=y_calc, y_true=y_true))
            metrics_writer.write('{}\n'.format(pbar))
        with tqdm(train_loader, desc='inference train', ascii=True) as pbar:
            inference(device, model, pbar,
                      lambda y_pred, y_calc, y_true: pbar.set_postfix(
                          y_pred=y_pred, y_calc=y_calc, y_true=y_true))
            metrics_writer.write('{}\n'.format(pbar))

        # update learning rate
        if isinstance(scheduler, O.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
            lr = '%.7f' % scheduler._last_lr[0]
        else:
            scheduler.step()
            lr = '%.7f' % scheduler.get_last_lr()[0]

        metrics_writer.flush()

        if TENSORBOARD:
            for name, weight in model.module.named_parameters():
                # check grad vanishing
                if name in (
                        'resnet50.base_model.4.0.conv1.weight',
                        'tsm_features.0.weight',
                        'projection1.projection.0.weight',
                        'projection2.projection.0.weight',
                        'trans1.trans_encoder.layers.0.linear1.weight') \
                                or re.match(r'pc1\.[a-z]+\.\d.weight', name) is not None:
                    tb.add_histogram(name, weight, epoch)
                    tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        # save model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if opt.local_rank == -1 else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }
        torch.save(checkpoint, opt.ckpt_save_path)

    metrics_writer.close()

    if TENSORBOARD:
        tb.close()


def run_train(opt):

    torch.manual_seed(888)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")
    model = RepNet(NUM_FRAMES, NUM_DMODEL).to(device)

    # data loader
    train_dataset = CountixDataset(DATASET_PREFIX, 'train')
    valid_dataset = CountixDataset(DATASET_PREFIX, 'val')
    test_dataset = CountixDataset(DATASET_PREFIX, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=1, shuffle=False)

    if opt.local_rank == -1:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=False)
    else:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                num_workers=4, drop_last=True, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size,
                num_workers=4, drop_last=True, sampler=valid_sampler)

        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                find_unused_parameters=True)

    # hyper parameters
    optimizer = O.Adam(model.parameters(), lr=0.001)
    # optimizer = O.SGD(model.parameters(), lr=lr)
    # scheduler = O.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # scheduler = O.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
    # scheduler = O.lr_scheduler.MultiStepLR(optimizer, milestones=[
    #         3, 10, 50, 100, 200, 300, 400], gamma=0.6)
    scheduler = O.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-7)
    criterions = [nn.SmoothL1Loss(), nn.BCEWithLogitsLoss()]

    train_loop(opt, model,
               train_loader, valid_loader, test_loader,
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
    parser.add_argument(
            '--data_root',
            default='/data/datasets/cv/countix',
            type=str,
            dest='data_root',
            help="")
    parser.add_argument(
            '--ckpt_from_path',
            default='last.pt',
            type=str,
            dest='ckpt_from_path',
            help="")
    parser.add_argument(
            '--ckpt_save_path',
            default='last.pt',
            type=str,
            dest='ckpt_save_path',
            help="")

    args = parser.parse_args()

    run_train(args)
