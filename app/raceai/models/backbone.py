#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision.models import resnet18


class Vgg16(object):
    def __init__(self, cfg):
        pass


class Resnet18(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.ckpt_path.startswith('file://'):
            ckpt_path = cfg.ckpt_path[7:]
        else:
            ckpt_path = cfg.ckpt_path
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, cfg.num_classes)
        self.model.load_state_dict(torch.load(ckpt_path))

    def forward(self, x):
        return self.model(x)
