#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file yolo_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-02-02 18:23

import traceback
import argparse
import cv2
import os
import torch
import numpy as np
from omegaconf import OmegaConf

from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import letterbox, LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.general import scale_coords
from yolov5.utils.plots import plot_one_box

from raceai.utils.misc import race_load_class, race_report_result
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)

import time # noqa
import zmq

view_debug = False
race_set_loglevel('info')

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))


def detect(opt):
    Logger.info('loading weight [%s]...' % opt.weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    t0 = time.time()
    model = attempt_load(opt.weights, map_location=device)
    imgsz = check_img_size(opt.img_size, s=model.stride.max())
    if half:
        model.half()
    t1 = time.time()
    Logger.info('loading finish [%.2fs]' % (t1 - t0))
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    model(img.half() if half else img)
    t2 = time.time()
    Logger.info('time consuming: [%.2fs]' % (t2 - t1))
    zmq_stats_count = 0
    while True:
        try:
            cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
            zmq_stats_count += 1
            stime = time.time()
            cfg = OmegaConf.create(cfg)
            Logger.info(cfg)
            if 'pigeon' not in cfg:
                continue
            conf_thres = opt.conf_thres
            iou_thres = opt.iou_thres
            msgkey = opt.topic
            if 'msgkey' in cfg.pigeon:
                msgkey = cfg.pigeon.msgkey
            resdata = {'pigeon': dict(cfg.pigeon), 'task': opt.topic, 'errno': 0, 'result': []}
            data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()
            for image_path, source in data_loader:
                if isinstance(image_path, str):
                    dataset = LoadImages(image_path, img_size=imgsz)
                    path, img, im0, _ = next(iter(dataset))
                    image_file = os.path.basename(source)
                else:
                    im0 = image_path
                    if im0 is None: # TODO
                        break
                    img = letterbox(im0, imgsz)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    image_file = f'{source}.jpg'
                resitem = {
                        'image_path': image_file,
                        'image_width': im0.shape[1],
                        'image_height': im0.shape[0],
                        'predict_box':[]}
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=opt.augment)[0]
                if 'nms' in cfg:
                    if 'conf_thres' in cfg.nms:
                        conf_thres = cfg.nms.conf_thres
                    if 'iou_thres' in cfg.nms:
                        iou_thres = cfg.nms.iou_thres
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                for i, det in enumerate(pred):
                    if len(det) == 0:
                        continue
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        resitem['predict_box'].append({
                            'label': int(cls),
                            'conf': '%.3f' % conf,
                            'xyxy': [int(x) for x in xyxy]})
                        if view_debug:
                            plot_one_box(xyxy, im0, label='%.3f' % conf, line_thickness=1)
                            cv2.imwrite(f'/raceai/data/{image_file}', im0)
                resdata['result'].append(resitem)
            resdata['running_time'] = round(time.time() - stime, 3)
            race_report_result(msgkey, resdata)
            Logger.info('[%6d] time consuming: [%.2f]s' % (zmq_stats_count % 99999, resdata['running_time']))
            Logger.info(resdata)
        except Exception:
            resdata['errno'] = -1 # todo
            resdata['traceback'] = traceback.format_exc()
            race_report_result(msgkey, resdata)
            Logger.error(resdata)
        time.sleep(0.01)


if __name__ == '__main__':
    Logger.info('start yolo main')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/ckpts/l.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--topic', default='zmq.yolov5l.inference', help='sub topic')
    opt = parser.parse_args()

    race_set_logfile('/tmp/raceai-{opt.topic}.log')

    Logger.info(opt)
    zmqsub.subscribe(opt.topic)
    race_report_result('add_topic', opt.topic)

    try:
        with torch.no_grad():
            Logger.info('start yolo detect')
            detect(opt)
            Logger.info('never run here')
    finally:
        race_report_result('del_topic', opt.topic)
