#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file pl_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-03-03 13:55


import traceback
import argparse
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from raceai.utils.misc import race_load_class, race_report_result
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)
from raceai.models.backbone import Resnet18
from raceai.runner.pl import PlClassifier

import time # noqa
import zmq


race_set_loglevel('info')
race_set_logfile('/tmp/raceai-pl.log')

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))


def classifer(opt):
    Logger.info('loading weight [%s]...' % opt.weights)

    bbmodel = Resnet18(OmegaConf.create('''{
        "device": "%s",
        "num_classes": %d,
        "weights": false}''' % (opt.device, opt.num_classes)))

    model = PlClassifier(bbmodel, None, None)
    ckpt = torch.load(opt.weights, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['state_dict'])
    model.eval() # ignore bn and dropout layers
    while True:
        try:
            cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
            stime = time.time()
            cfg = OmegaConf.create(cfg)
            Logger.info(cfg)
            if 'pigeon' not in cfg:
                continue
            msgkey = opt.topic
            if 'msgkey' in cfg.pigeon:
                msgkey = cfg.pigeon.msgkey
            resdata = {'pigeon': dict(cfg.pigeon), 'task': opt.topic, 'errno': 0, 'result': []}
            data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()
            for images, labels, paths in data_loader:
                y_preds = model(images)
                print(y_preds)
                y_preds = F.softmax(y_preds, dim=1)
                for path, tag, y_pred in list(zip(paths, labels, y_preds)):
                    if isinstance(tag, torch.Tensor):
                        tag = tag.cpu().item()
                    probs = y_pred.cpu()
                    probs_sorted = probs.sort(descending=True)
                    result = {
                        'image_path': path,
                        'image_id': tag,
                        'probs': probs.numpy().astype(float).tolist(),
                        'probs_sorted': {
                            'values': probs_sorted.values.numpy().astype(float).tolist(),
                            'indices': probs_sorted.indices.numpy().astype(int).tolist()
                        }
                    }
                    resdata['result'].append(result)
            resdata['running_time'] = round(time.time() - stime, 3)
            Logger.info('time consuming: [%.2f]s' % (resdata['running_time']))
            race_report_result(msgkey, resdata)
            print(resdata)
        except Exception:
            resdata['errno'] = -1 # todo
            resdata['traceback'] = traceback.format_exc()
            race_report_result(msgkey, resdata)
            Logger.error(resdata)
        time.sleep(0.01)


if __name__ == '__main__':
    Logger.info('start pl main')

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='using cuda or cpu')
    parser.add_argument('--weights', type=str, default='/ckpts/best.pth', help='model.pt path(s)')
    parser.add_argument('--num_classes', type=int, default=10, help='inference size (pixels)')
    parser.add_argument('--topic', default='zmq.cr.resnet18.inference', help='sub topic')
    opt = parser.parse_args()
    Logger.info(opt)

    zmqsub.subscribe(opt.topic)

    race_report_result('add_topic', opt.topic)

    try:
        with torch.no_grad():
            Logger.info('start pl')
            classifer(opt)
            Logger.info('never run here')
    finally:
        race_report_result('del_topic', opt.topic)

# test
# cfg = '''{
#     "pigeon": {
#         "msgkey": "zmq.pl.test",
#         "user": "1",
#         "uuid": "100"
#     },
#     "data": {
#         "class_name": "raceai.data.process.PathListDataLoader",
#         "params": {
#             "data_source": [
#                 "cable/cable_1.jpg",
#                 "other/paper_1.jpg",
#                 "ring/ring_1.jpg"
#             ],
#             "dataset": {
#                 "class_name": "raceai.data.PredictListImageDataset",
#                 "params": {
#                     "data_prefix": "/raceai/data/datasets/cleaner_robot/imgs/",
#                     "input_size": 244
#                 }
#             },
#             "sample": {
#                 "batch_size": 32,
#                 "num_workers": 4,
#             }
#         }
#     }
# }'''
#
# with torch.no_grad():
#     s0 = time.time()
#     cfg = OmegaConf.create(cfg)
#     data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()
#     bbmodel = Resnet18(OmegaConf.create('{"device": "gpu", "num_classes": 3, "weights": false}'))
#     model = PlClassifier(bbmodel, None, None)
#     ckpt = torch.load("/raceai/data/ckpts/cleaner_robot/pl_resnet18_acc90.pth", map_location=lambda storage, loc: storage)
#     model.load_state_dict(ckpt['state_dict'])
#     s1 = time.time()
#     # preds = []
#     # preds.append(list(zip(paths, y_trues, y_preds)))
#     for images, labels, paths in data_loader:
#         y_preds = model(images)
#         y_preds = F.softmax(y_preds, dim=1)
#         for path, tag, y_pred in list(zip(paths, labels, y_preds)):
#             if isinstance(tag, torch.Tensor):
#                 tag = tag.cpu().item()
#             probs = y_pred.cpu()
#             probs_sorted = probs.sort(descending=True)
#             result = {
#                 'image_path': path,
#                 'image_id': tag,
#                 'probs': probs.numpy().astype(float).tolist(),
#                 'probs_sorted': {
#                     'values': probs_sorted.values.numpy().astype(float).tolist(),
#                     'indices': probs_sorted.indices.numpy().astype(int).tolist()
#                 }
#             }
#             print(result)
#     et = time.time()
#     print(et - s1, et - s0)
