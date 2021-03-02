#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import torch
import base64 # noqa
import io # noqa
import cv2 # noqa
import shutil
import PIL.Image as Image # noqa
import matplotlib.pyplot as plt # noqa

from omegaconf import OmegaConf
from torchvision import transforms
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

from raceai.utils.registrable import FunctionRegister
from raceai.utils.error import catch_error
from raceai.utils.misc import race_load_class
from raceai.runner.pl import PlClassifier, PlTrainer


@catch_error
@FunctionRegister.register('cls.inference')
def image_classifier_test(cfg):
    # Data
    data_loader_class = race_load_class(cfg.data.class_name)
    data_loader = data_loader_class(cfg.data.params).get()

    # Model
    test_model = race_load_class(cfg.model.class_name)(cfg.model.params)

    # Test
    test_model.eval()
    with torch.no_grad():
        results = []
        for inputs in data_loader:
            outputs = test_model(inputs)
            results.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        return {'errno': 0, 'result': results}


@catch_error
@FunctionRegister.register('cls.inference.pl')
def image_classifier_test_pl(cfg):
    # Data
    data_loader_class = race_load_class(cfg.data.class_name)
    data_loader = data_loader_class(cfg.data.params).get()

    # Model
    bbmodel = race_load_class(cfg.model.class_name)(cfg.model.params)

    # Test
    trainer = PlTrainer(False, None, **cfg.trainer)
    classifer = PlClassifier(bbmodel, None, None)
    result = trainer.predict(classifer, data_loader)
    print(result)
    return {'errno': 0, 'result': result[0]['output']}


@catch_error
@FunctionRegister.register('det.inference')
def image_detection_test(cfg):
    # Data
    data_loader = race_load_class(cfg.data.class_name)(cfg.cache_dir, cfg.data.params)
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


@catch_error
@FunctionRegister.register('det.features')
def image_detection_features(cfg):
    # Data
    data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()

    # Model
    model = race_load_class(cfg.model.class_name)(cfg.model.params)

    # Trainer
    trainer = race_load_class(cfg.trainer.class_name)

    results = {}
    for img, path in data_loader:
        fname = os.path.basename(path).split('.')[0]
        results[fname] = trainer(model, path, **cfg.trainer.params)

    return {'errno': 0, 'result': results}


@catch_error
@FunctionRegister.register('gan.faceswap')
def image_face_swap_gan(cfg):
    # Data
    # data_loader_class = race_load_class(cfg.data.class_name)
    # data_loader = data_loader_class(cfg.data.params).get()

    # Model
    output_path = race_load_class(cfg.model.class_name)(**cfg.model.params)

    # Get first frame
    b64data = ''
    videoCapture = cv2.VideoCapture(output_path)
    success, frame = videoCapture.read()
    if success:
        image = Image.fromarray(frame).convert('RGB')
        bio = io.BytesIO()
        image.save(bio, "PNG")
        bio.seek(0)
        b64data = base64.b64encode(bio.read()).decode()
    shutil.rmtree(os.path.dirname(output_path))
    return {'errno': 0, 'result': {
        'b64img': b64data}}


@catch_error
@FunctionRegister.register('nst.inference')
def neural_style_transfer(cfg):
    # Data
    data_loader_class = race_load_class(cfg.data.class_name)
    data_loader = data_loader_class(cfg.data.params).get()

    # Model
    style_model = race_load_class(cfg.model.class_name)(cfg.model.params)

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))])

    style_model.eval()
    with torch.no_grad():
        img, _ = next(data_loader)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = content_transform(img)
        out = style_model(img.unsqueeze(0)).cpu()
        out = out[0].clone().clamp(0, 255).numpy()
        out = out.transpose(1, 2, 0).astype("uint8")
        out = Image.fromarray(out)
        bio = io.BytesIO()
        out.save(bio, "PNG")
        bio.seek(0)
        return {'errno': 0, 'result': {
            'b64img': base64.b64encode(bio.read()).decode()}}


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
        config['cfg']['cache_dir'] = args.cache_dir
        runner = FunctionRegister.get_runner(config['task'])
        cfg = OmegaConf.create(config['cfg'])

    print(runner(cfg))
