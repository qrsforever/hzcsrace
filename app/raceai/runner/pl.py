#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file pl.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-27 13:09

import numpy as np # noqa
import torch # noqa
import torchvision # noqa
import pytorch_lightning as pl


class PlClassifier(pl.LightningModule):
    def __init__(self, trainer, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.trainer = trainer
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)

    def criterion(self, inputs, targets):
        return torch.nn.functional.cross_entropy(inputs, targets)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        inputs, y_trues, paths = batch
        y_preds = self(inputs)
        loss = self.criterion(y_preds, y_trues)
        acc = (torch.argmax(y_preds, dim=1) == y_trues).float().mean()
        log = {'loss': loss, 'acc': acc}
        return log

    def training_epoch_end(self, outputs):
        log = {
            'train_loss': torch.stack([x['loss'] for x in outputs]).mean()
        }
        if 'acc' in outputs[0]:
            log['train_acc'] = torch.stack([x['acc'] for x in outputs]).mean()
        self.log_dict(log, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        inputs, y_trues, paths = batch
        y_preds = self(inputs)
        loss = self.criterion(y_preds, y_trues)
        acc = (torch.argmax(y_preds, dim=1) == y_trues).float().mean()
        log = {'val_loss': loss, 'val_acc': acc}
        return log

    def validation_epoch_end(self, outputs):
        log = {
            'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean()
        }
        if 'val_acc' in outputs[0]:
            log['val_acc'] = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log_dict(log, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, y_trues, paths = batch
        y_preds = self(inputs)
        acc = (torch.argmax(y_preds, dim=1) == y_trues).float().mean()
        log = {'test_acc': acc}
        return log

    def test_epoch_end(self, outputs):
        log = {}
        if 'test_acc' in outputs[0]:
            log['test_acc'] = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log_dict(log, prog_bar=True)

    def get_progress_bar_dict(self):               
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def fit(self, train_loader, valid_loader):
        return self.trainer.fit(self, train_loader, valid_loader)

    def test(self, test_loader):
        return self.trainer.test(self, test_loader)


class PlTrainer(pl.Trainer):
    def __init__(self, cfg):
        super().__init__(**cfg)
