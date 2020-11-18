#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision.models import resnet18
from raceai.utils.misc import race_prepare_weights


class Vgg16(object):
    def __init__(self, cfg):
        pass


class Resnet18(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        weights = race_prepare_weights(cfg.weights)
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, cfg.num_classes)
        self.model.load_state_dict(torch.load(weights))
        self.use_gpu = True if cfg.device == 'cuda' else False
        if self.use_gpu:
            self.model.cuda()

    def forward(self, x):
        if self.use_gpu:
            x = x.cuda()
        return self.model(x)
