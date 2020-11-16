#!/usr/bin/python3
# -*- coding: utf-8 -*-


import yaml
from yacs.config import CfgNode
from detectron2 import model_zoo
from detectron2.config import get_cfg

from raceai.utils.misc import race_convert_dictkeys
from raceai.utils.misc import race_prepare_weights


class DetBaseModel(object):
    def __init__(self, cfg):
        cfg.weights = race_prepare_weights(cfg.weights)
        cfg = race_convert_dictkeys(cfg, uppercase=True)
        self.cfg = CfgNode.load_cfg(yaml.dump(cfg))

    def merge_cfg(self, yaml_file):
        _cfg = get_cfg()
        _cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
        _cfg.MODEL.merge_from_other_cfg(self.cfg)
        return _cfg


class MaskRCNN(DetBaseModel):
    pass
