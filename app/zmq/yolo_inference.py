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
import torch
from omegaconf import OmegaConf

from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.general import scale_coords
from yolov5.utils.plots import plot_one_box

from raceai.utils.misc import race_load_class, race_report_result
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)

import time # noqa
import zmq

view_debug = False
race_set_loglevel('info')
race_set_logfile('/tmp/raceai-yolo.log')

context = zmq.Context()
topic = 'zmq.yolo.inference'
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))
zmqsub.subscribe(topic)


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
    model(img.half() if half else img) if device.type != 'cpu' else None
    t2 = time.time()
    Logger.info('time consuming: [%.2fs]' % (t2 - t1))
    while True:
        try:
            cfg = ''.join(zmqsub.recv_string().split(' ')[1:])
            cfg = OmegaConf.create(cfg)
            Logger.info(cfg)
            if 'pigeon' not in cfg:
                continue
            msgkey = f'{topic}.result'
            if 'msgkey' in cfg.pigeon:
                msgkey = cfg.pigeon.msgkey
            resdata = {'pigeon': dict(cfg.pigeon), 'task': topic, 'errno': 0, 'result': []}
            t3 = time.time()
            data_loader = race_load_class(cfg.data.class_name)(cfg.data.params).get()
            for source in data_loader:
                dataset = LoadImages(source, img_size=imgsz)
                path, img, im0, _ = next(iter(dataset))
                Logger.info(path)
                resitem = {'image_path': path, 'faces_det':[]}
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=opt.augment)[0]
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
                for i, det in enumerate(pred):
                    if len(det) == 0:
                        continue
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, _ in reversed(det):
                        resitem['faces_det'].append({
                            'conf': '%.3f' % conf,
                            'xyxy': [int(x) for x in xyxy]})
                        if view_debug:
                            plot_one_box(xyxy, im0, label='%.3f' % conf, line_thickness=1)
                            cv2.imwrite(f'/raceai/data/{i}.png', im0)
                resdata['result'].append(resitem)
            Logger.info(resdata)
            race_report_result(msgkey, resdata)
        except Exception:
            resdata['errno'] = -1 # todo
            resdata['traceback'] = traceback.format_exc()
            race_report_result(msgkey, resdata)
            Logger.error(resdata)
        Logger.info('time consuming: [%.2f]s' % (time.time() - t3))
        time.sleep(0.01)


if __name__ == '__main__':
    Logger.info('start yolo main')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/ckpts/l.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    Logger.info(opt)

    with torch.no_grad():
        Logger.info('start yolo detect')
        detect(opt)
        Logger.info('never run here')
