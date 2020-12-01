#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file pl.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-27 13:09

import os
import numpy as np # noqa
import torch # noqa
import torchvision # noqa
import pytorch_lightning as pl
from torch.nn import functional as F


class PlClassifier(pl.LightningModule):
    def __init__(self, trainer, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.trainer = trainer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = {
            "train": [],
            "valid": []
        }

    def _get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def forward(self, x):
        return self.model(x)

    def criterion(self, inputs, targets):
        return torch.nn.functional.cross_entropy(inputs, targets)

    def configure_optimizers(self):
        # return [self.optimizer], [self.scheduler]
        return {'monitor': 'val_loss', 'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx):
        inputs, y_trues, paths = batch
        y_preds = self(inputs)
        loss = self.criterion(y_preds, y_trues)
        acc = (torch.argmax(y_preds, dim=1) == y_trues).float().mean()
        log = {'loss': loss, 'acc': acc}
        return log

    def training_epoch_end(self, outputs):
        log = {
            'lr': self._get_lr(),
            'train_loss': torch.stack([x['loss'] for x in outputs]).mean()
        }
        if 'acc' in outputs[0]:
            log['train_acc'] = torch.stack([x['acc'] for x in outputs]).mean()
        # self.metrics['train'].append(log)
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
        # log['early_stop_on'] = log['val_loss']
        # log['checkpoint_on'] = log['val_loss']
        # self.metrics['valid'].append(log)
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
            log['test_acc'] = torch.stack([x['test_acc'] for x in outputs]).mean().numpy()
        self.log_dict(log)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def fit(self, train_loader, valid_loader):
        return self.trainer.fit(self, train_loader, valid_loader)

    def test(self, test_loader):
        return self.trainer.test(self, test_loader)

    def predict(self, test_loader):
        def predict_step(self, batch, batch_idx):
            inputs, _, paths = batch
            y_preds = self(inputs)
            return list(zip(paths, F.softmax(y_preds, dim=1)))

        def predict_epoch_end(self, outputs):
            result = {'output':[]}
            for item in outputs:
                for path, preds in item:
                    fname = os.path.basename(path)
                    probs = preds.cpu().numpy().astype(float).tolist()
                    result['output'].append({'fname': fname, 'probs': probs})
            self.log_dict(result)
        try:
            _test_step = getattr(self.__class__, 'test_step', None)
            _test_epoch_end = getattr(self.__class__, 'test_epoch_end', None)
            setattr(self.__class__, 'test_step', predict_step)
            setattr(self.__class__, 'test_epoch_end', predict_epoch_end)
            return self.trainer.test(self, test_loader, verbose=True)
        except Exception as err:
            raise err
        finally:
            if _test_step:
                setattr(self.__class__, 'test_step', _test_step)
            if _test_epoch_end:
                setattr(self.__class__, 'test_epoch_end', _test_epoch_end)


class PlTrainer(pl.Trainer):
    def __init__(self, logger, callbacks, *args, **kwargs):
        super().__init__(
                logger=logger,
                callbacks=callbacks,
                num_sanity_val_steps=0, *args, **kwargs)
