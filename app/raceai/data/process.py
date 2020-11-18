#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import base64
from raceai.utils.misc import race_load_class
from torch.utils.data import DataLoader


class BaseDataLoader(object):
    dataset = None 

    def get_testloader(self):
        return DataLoader(self.dataset)


class Base64DataLoader(BaseDataLoader):
    def __init__(self, tmp_dir, cfg):
        imgpath = os.path.join(tmp_dir, 'b4img.png')
        with open(imgpath, 'wb') as fout:
            fout.write(base64.b64decode(cfg.data_source))
        self.dataset = race_load_class(cfg.dataset.class_name)(imgpath, **cfg.dataset.params)


class PathListDataLoader(BaseDataLoader):
    def __init__(self, tmp_dir, cfg):
        imgpaths = [str(p) for p in cfg.data_source]
        self.dataset = race_load_class(cfg.dataset.class_name)(imgpaths, **cfg.dataset.params)
