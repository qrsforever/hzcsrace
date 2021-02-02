#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file yolo_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-02-02 18:23

import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from omegaconf import OmegaConf
from numpy import random

from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression # noqa
from raceai.utils.misc import race_load_class

import time # noqa
import zmq

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))
zmqsub.subscribe('zmq.yolo.inference')


def detect(opt):
    print('Loading weight [%s]...' % opt.weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    t0 = time.time()
    model = attempt_load(opt.weights, map_location=device)
    imgsz = check_img_size(opt.img_size, s=model.stride.max())
    if half:
        model.half()
    t1 = time.time()
    print('Loading finish [%.2fs]' % (t1 - t0))
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # noqa
    t2 = time.time()
    print('Use time: [%.2fs]' % (t2 - t1))
    while True:
        try:
            cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
            cfg = OmegaConf.create(cfg)
            t3 = time.time()
            data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()
            for source in data_loader:
                dataset = LoadImages(source, img_size=imgsz)
                path, img, im0s, vid_cap = next(iter(dataset))
                print(path)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float() 
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=opt.augment)[0]
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
                print(pred)
        except Exception as err:
            print(err)
        print('Time: [%.2f]s' % (time.time() - t3))
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/raceai/data/ckpts/yolov5/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)
