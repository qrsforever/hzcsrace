#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from abc import ABC, abstractmethod # noqa
from torchvision.models import resnet18
from raceai.utils.misc import race_prepare_weights


class BBM(ABC, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bbmodel = self.tl_model(cfg.num_classes, cfg.weights)
        self.use_gpu = True if cfg.device == 'cuda' else False
        if self.use_gpu:
            self.bbmodel.cuda()

    def use_gpu(self):
        return self.use_gpu

    def forward(self, x):
        if self.use_gpu:
            x = x.cuda()
        return self.bbmodel(x)

    @abstractmethod
    def tl_model(self, num_classes, weights):
        """
        """


class Resnet18(BBM):
    def tl_model(self, num_classes, weights): # transfer learning
        pretrained = False
        if isinstance(weights, bool):
            pretrained = weights
        model = resnet18(pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if isinstance(weights, str):
            ckpt = torch.load(race_prepare_weights(weights))
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            model.load_state_dict(ckpt)
        return model
