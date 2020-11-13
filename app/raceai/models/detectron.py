#!/usr/bin/python3
# -*- coding: utf-8 -*-


class MaskRCNN(object):
    def __init__(self, cfg):
        super().__init__()
        if cfg.ckpt_path.startswith('file://'):
            ckpt_path = cfg.weights[7:]
        else:
            ckpt_path = cfg.weights
