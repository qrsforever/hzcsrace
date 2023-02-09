#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import base64
import omegaconf
from torch.utils.data import DataLoader
from raceai.utils.misc import race_load_class


def to_list_dict(data):
    if isinstance(data, omegaconf.ListConfig):
        sources = list(data)
        for i in range(len(sources)):
            sources[i] = to_list_dict(sources[i])
    elif isinstance(data, omegaconf.DictConfig):
        sources = dict(data)
    else:
        return data
    return sources


class BaseDataLoader(object):
    dataset = None

    def __init__(self, sources, cfg):
        sources = to_list_dict(sources)
        self.dataset = race_load_class(cfg.dataset.class_name)(sources, cfg.dataset.params)
        if 'sample' in cfg:
            self.dataloader = DataLoader(self.dataset, **cfg.sample)
        else:
            self.dataloader = iter(self.dataset)

    def get(self):
        return self.dataloader

    def clr(self):
        pass


class Base64DataLoader(BaseDataLoader):
    def __init__(self, cfg):
        # TODO resource release
        data = cfg.data_source.split(',')
        suffix = 'png'
        if len(data) == 2:
            if data[0].find('wav') > 0:
                suffix = 'wav'
            elif data[0].find('flac') > 0:
                suffix = 'flac'
        filepath = os.path.join('/tmp', 'b64_%d.%s' % (1000 * time.time(), suffix))
        with open(filepath, 'wb') as fout:
            fout.write(base64.b64decode(data[-1]))
        super().__init__(filepath, cfg)
        self.filepath = filepath

    def clr(self):
        os.remove(self.filepath)


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
        self.file_paths = file_paths

    def clr(self):
        for path in self.file_paths:
            os.remove(path)
        

class ListBase64DataLoader(BaseDataLoader):
    def __init__(self, cfg):
        file_paths = []
        for b64str in cfg.data_source:
            imgpath = os.path.join('/tmp', 'b64_%d.png' % (1000 * time.time()))
            with open(imgpath, 'wb') as fout:
                fout.write(base64.b64decode(b64str.split(',')[-1]))
            file_paths.append(imgpath)
        super().__init__(file_paths, cfg)
        self.file_paths = file_paths

    def clr(self):
        for path in self.file_paths:
            os.remove(path)


class PathListDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        super().__init__(cfg.data_source, cfg)


class VideoDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        super().__init__(cfg.data_source, cfg)


class LocalDataLoader(BaseDataLoader):
    def __init__(self, cfg):
        super().__init__(cfg.data_source, cfg)
