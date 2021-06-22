#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import time
import zmq
import shutil

import torch

from collections import Callable
from sense.controller import Controller
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.loading import build_backbone_network
from sense.loading import load_backbone_model_from_config

from omegaconf import OmegaConf
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)
from raceai.utils.misc import (
        race_oss_client,
        race_object_put,
        race_report_result, 
        race_data)


race_set_loglevel('info')
race_set_logfile('/tmp/raceai-sense.log')

_DEBUG_ = True

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

osscli = race_oss_client(bucket_name='raceai')

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='Sense Config')
parser.add_argument('--topic', type=str, required=True, help='zmq topic')
parser.add_argument('--custom_classifier', type=str, required=True, help='classifer ckpts')

args = parser.parse_args()


def _report_result(msgkey, resdata):
    if not _DEBUG_:
        race_report_result(msgkey, resdata)
        race_report_result('zmp_run', f'{args.topic}:120')
    else:
        pass


class ResultCallback(Callable):
    def __call__(self, out):
        return True


def inference(model, opt):
    msgkey = args.topic
    if 'msgkey' in opt.pigeon:
        msgkey = opt.pigeon.msgkey
    else:
        opt.pigeon.msgkey = msgkey

    user_code = 'unkown'
    if 'user_code' in opt.pigeon:
        user_code = opt.pigeon.user_code

    outputpath = os.path.join('/outputs', user_code)
    shutil.rmtree(outputpath, ignore_errors=True)
    os.makedirs(outputpath, exist_ok=True)
    args.outputpath = outputpath

    if 'save_video' in opt:
        path_out = os.path.join(outputpath, 'sense-target.mp4')
    else:
        path_out = None

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4)
    ]

    controller = Controller(
        neural_network=model,
        post_processors=postprocessor,
        results_display=None,
        callbacks=[ResultCallback()],
        path_in=race_data(opt.video),
        path_out=path_out,
        use_gpu=True,
    )
    controller.run_inference()


if __name__ == "__main__":

    if not _DEBUG_:
        zmqsub.subscribe(args.topic)
        race_report_result('add_topic', args.topic)

    try:
        backbone_model_config, backbone_weights = load_backbone_model_from_config(args.custom_classifier)
        checkpoint_classifier = torch.load(os.path.join(args.custom_classifier, 'best_classifier.checkpoint'))
        backbone_network = build_backbone_network(backbone_model_config, backbone_weights,
                                                  weights_finetuned=checkpoint_classifier)
        with open(os.path.join(args.custom_classifier, 'label2int.json')) as file:
            class2int = json.load(file)
        INT2LAB = {value: key for key, value in class2int.items()}

        gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                                num_out=len(INT2LAB))
        gesture_classifier.load_state_dict(checkpoint_classifier)
        gesture_classifier.eval()

        model = Pipe(backbone_network, gesture_classifier)

        if not _DEBUG_:
            while True:
                Logger.info('wait task')
                race_report_result('zmp_end', args.topic)
                zmq_cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
                race_report_result('zmp_run', f'{args.topic}:30')
                zmq_cfg = OmegaConf.create(zmq_cfg)
                Logger.info(zmq_cfg)
                if 'pigeon' not in zmq_cfg:
                    continue
                Logger.info(zmq_cfg.pigeon)
                inference(model, zmq_cfg)
                time.sleep(0.01)
        else:
            zmq_cfg = {
                "pigeon": {"msgkey": "123"},
                "video": "https://raceai.s3.didiyunapi.com/data/media/videos/sense_test.mp4"
            }
            zmq_cfg = OmegaConf.create(zmq_cfg)
            Logger.info(zmq_cfg)
            inference(model, zmq_cfg)
    finally:
        if _DEBUG_:
            race_report_result('del_topic', args.topic)
        Logger.info('end')
