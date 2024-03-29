#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tempfile
import torch
from raceai.utils.error import catch_error
from raceai.utils.misc import race_load_class

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode

# cfg
# {
#     "task": "cls.test",
#     "cfg": {
#         "general": {
#             "device": "cuda",
#             "work_dir": f"/raceai/data/users/{userid}/{projectid}"
#         },
#         "data": {
#             "class_name": "raceai.data.process.Base64DataLoader",
#             "params": {
#                 "data_name": dataset,
#                 "data_source": b4data,
#                 "dataset": {
#                     "class_name": "raceai.data.PredictSingleImageDataset",
#                     "params": {
#                         "input_size": 224,
#                         "mean": [0.6535,0.6132,0.5643],
#                         "std": [0.2165,0.2244,0.2416]
#                     }
#                 }
#             }
#         },
#         "model": {
#             "class_name": f"raceai.models.backbone.{model}",
#             "params": {
#                 "num_classes": 4,
#                 "ckpt_path": f"file:///raceai/data/ckpts/{dataset}_{model}.pth"
#             }
#         }
#     }
# }


@catch_error
def image_classifier_test(cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Data
        data_loader = race_load_class(cfg.data.class_name)(tmp_dir, cfg.data.params)
        test_loader = data_loader.get_testloader()

        # Model
        use_gpu = True if cfg.general.device == 'cuda' else False
        test_model = race_load_class(cfg.model.class_name)(cfg.model.params)
        if use_gpu:
            test_model.cuda()

        # Test
        test_model.eval()
        with torch.no_grad():
            results = []
            for inputs in test_loader:
                if use_gpu:
                    inputs = inputs.cuda()
                outputs = test_model(inputs)
                results.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            return {'errno': 0, 'result': results}


@catch_error
def image_dectection_test(cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Data
        data_loader = race_load_class(cfg.data.class_name)(tmp_dir, cfg.data.params)
        test_loader = data_loader.get_testloader()

        cfg = get_cfg()

        cfg.merge_from_file(
          model_zoo.get_config_file(
            'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
          )
        )
