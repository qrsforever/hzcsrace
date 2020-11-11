#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import base64
from raceai.utils.misc import race_load_class
from torch.utils.data import DataLoader


class Base64DataLoader(object):
    def __init__(self, tmp_dir, cfg):
        img_path = os.path.join(tmp_dir, 'b4img.png')
        with open(img_path, 'wb') as fout:
            fout.write(base64.b64decode(cfg.data_source))
        self.dataset = race_load_class(cfg.dataset.class_name)(tmp_dir, **cfg.dataset.params)

    def get_testloader(self):
        return DataLoader(self.dataset)
