#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import base64
from torch.utils.data import DataLoader
from raceai.utils.misc import race_load_class
# from raceai.utils.misc import race_data


class BaseDataLoader(object):
    dataset = None

    def __init__(self, filepath, cfg):
        self.dataset = race_load_class(cfg.dataset.class_name)(filepath, cfg.dataset.params)
        if 'sample' in cfg:
            self.dataloader = DataLoader(self.dataset, **cfg.sample)
        else:
            self.dataloader = iter(self.dataset)

    def get(self):
        return self.dataloader


class Base64DataLoader(BaseDataLoader):
    def __init__(self, cfg):
        imgpath = os.path.join('/tmp/', 'b4img_%d.png' % time.time())
        with open(imgpath, 'wb') as fout:
            fout.write(base64.b64decode(cfg.data_source.split(',')[-1]))
        super().__init__(imgpath, cfg)


class JsonBase64DataLoader(BaseDataLoader):
    def __init__(self, cfg):
        file_paths = []
        for imgpath, b64str in cfg.data_source.items():
            imgdir = os.path.dirname(imgpath)
            if not os.path.exists(imgdir):
                os.makedirs(imgdir, exist_ok=True)
            with open(imgpath, 'wb') as fout:
                fout.write(base64.b64decode(b64str.split(',')[-1]))
            file_paths.append(imgpath)
        super().__init__(file_paths, cfg)


class PathListDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        # imgpaths = []
        # for item in cfg.data_source:
        #     if isinstance(item, str):
        #         pass
        #     elif isinstance(item, dict):
        #         pass
        # imgpaths = [race_data(str(p)) for p in cfg.data_source]
        super().__init__(cfg.data_source, cfg)


class LocalDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        super().__init__(cfg.data_source, cfg)
