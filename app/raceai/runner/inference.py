#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import json
import torch
import base64 # noqa
import io # noqa
import cv2 # noqa
import PIL.Image as Image # noqa
import matplotlib.pyplot as plt # noqa

from omegaconf import OmegaConf
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

from raceai.utils.registrable import FunctionRegister
from raceai.utils.error import catch_error
from raceai.utils.misc import race_load_class

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
#                 "weights": f"file:///raceai/data/ckpts/{dataset}_{model}.pth"
#             }
#         }
#     }
# }


@FunctionRegister.register('cls.inference')
@catch_error
def image_classifier_test(cfg):
    # Data
    data_loader = race_load_class(cfg.data.class_name)(cfg.runner.cache_dir, cfg.data.params)
    test_loader = data_loader.get_testloader()

    # Model
    test_model = race_load_class(cfg.model.class_name)(cfg.runner.device, cfg.model.params)

    # Test
    test_model.eval()
    with torch.no_grad():
        results = []
        for inputs in test_loader:
            outputs = test_model(inputs)
            results.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        return {'errno': 0, 'result': results}


# {
#     "task": "det.test",
#     "cfg": {
#         "runner": {
#             "device": 'cuda',
#             "yaml_file": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
#         },
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
#                 "weights": f"/raceai/data/ckpts/{dataset}_{model}.pth",
#                 "roi_heads": {
#                     "score_thresh_test": 0.85,
#                     "num_classes": 1
#                 }
#             }
#         }
#     }
# }

@FunctionRegister.register('det.inference')
@catch_error
def image_detection_test(cfg):
    # Data
    data_loader = race_load_class(cfg.data.class_name)(cfg.runner.cache_dir, cfg.data.params)
    test_loader = data_loader.get_testloader()

    # Model
    test_model = race_load_class(cfg.model.class_name)(cfg.model.params)
    _cfg = test_model.merge_cfg(cfg.runner.yaml_file)
    _cfg.OUTPUT_DIR = cfg.runner.cache_dir

    # Test
    im = test_loader.dataset[0]
    outputs = DefaultPredictor(_cfg)(im)

    # Result
    v = Visualizer(
        im[:, :, ::-1],
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )

    instances = outputs['instances'].to('cpu')
    if instances.has('pred_masks'):
        instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)

    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()

    result = 'ok'
    # img = Image.fromarray(v.get_image())
    # img.save('/raceai/tmp/test.png')
    # bio = io.BytesIO()
    # img.save(bio, "PNG")
    # bio.seek(0)
    # result = {
    #     'b64img': base64.b64encode(bio.read()).decode(),
    # }
    return {'errno': 0, 'result': result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--cache_dir',
            default='/tmp',
            type=str,
            dest='cache_dir',
            help="cache dir")
    parser.add_argument(
            '--config',
            default='0.0.0.0',
            type=str,
            dest='config',
            help="json configure file")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
        config['cfg']['runner']['cache_dir'] = args.cache_dir
        runner = FunctionRegister.get_caller(config['task'])
        cfg = OmegaConf.create(config['cfg'])

    print(runner(cfg))
