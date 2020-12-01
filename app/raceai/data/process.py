#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import base64
from raceai.utils.misc import race_load_class
from raceai.utils.misc import race_data
from torch.utils.data import DataLoader


class BaseDataLoader(object):
    dataset = None

    def __init__(self, filepath, cfg):
        self.dataset = race_load_class(cfg.dataset.class_name)(filepath, cfg.dataset.params)
        self.dataloader = DataLoader(self.dataset, **cfg.sample)

    def get(self):
        return self.dataloader


class Base64DataLoader(BaseDataLoader):
    def __init__(self, cfg):
        imgpath = os.path.join('/tmp/', 'b4img_%s.png' % cfg.data_source[:6])
        with open(imgpath, 'wb') as fout:
            fout.write(base64.b64decode(cfg.data_source))
        super().__init__(imgpath, cfg)


class PathListDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        imgpaths = [race_data(str(p)) for p in cfg.data_source]
        super().__init__(imgpaths, cfg)


class LocalDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        super().__init__(cfg.data_source, cfg)
