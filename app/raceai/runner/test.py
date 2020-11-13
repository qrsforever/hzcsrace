#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tempfile
import torch
import base64
import io
import PIL.Image as Image
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
        use_gpu = True if cfg.model.device == 'cuda' else False
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

# {
#     "task": "det.test",
#     "cfg": {
#         "runner": {
#             "yaml_file": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
#         }
#         "general": {
#             "work_dir": f"/raceai/data/users/{userid}/{projectid}"
#         },
#         "data": {
#             "class_name": "raceai.data.process.Base64DataLoader",
#             "params": {
#                 "data_name": dataset,
#                 "data_source": b4data,
#                 "dataset": {
#                     "class_name": "raceai.data.PredictSingleImageRaw"
#                 }
#             }
#         },
#         "model": {
#             "class_name": f"raceai.models.detectron.{model}",
#             "params": {
#                 "DEVICE": 'cuda',
#                 "WEIGHTS": f"file:///raceai/data/ckpts/{dataset}_{model}.pth",
#                 "ROI_HEADS": {
#                     "SCORE_THRESH_TEST": 0.85,
#                     "NUM_CLASSES": 1
#                 }
#             }
#         }
#     }
# }


@catch_error
def image_dectection_test(cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Data
        data_loader = race_load_class(cfg.data.class_name)(tmp_dir, cfg.data.params)
        test_loader = data_loader.get_testloader()

        # Model
        # test_model = race_load_class(cfg.model.class_name)(cfg.model.params)

        # Cfg
        _cfg = get_cfg()
        _cfg.merge_from_file(model_zoo.get_config_file(cfg.runner.yaml_file))
        _cfg.OUTPUT_DIR = tmp_dir

        _cfg.MODEL.WEIGHTS = cfg.model.params.WEIGHTS
        _cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.model.params.ROI_HEADS.SCORE_THRESH_TEST
        _cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.model.params.ROI_HEADS.NUM_CLASSES

        im = next(iter(test_loader))
        outputs = DefaultPredictor(_cfg)(im)

        v = Visualizer(
            im[:, :, ::-1],
            scale=1.,
            instance_mode=ColorMode.IMAGE
        )

        instances = outputs['instances'].to('cpu')
        if instances.has('pred_masks'):
            instances.remove('pred_masks')
        v = v.draw_instance_predictions(instances)

        img = Image.fromarray(v.get_image())
        bio = io.BytesIO()
        img.save(bio, "PNG")
        bio.seek(0)
        result = base64.b64encode(bio.read()).decode()
        return result
