#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file ccpd2019.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-04-09 20:12


import traceback
import argparse
import cv2
import torch
from PIL import Image
from omegaconf import OmegaConf

from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.general import scale_coords
from yolov5.utils.plots import plot_one_box

from torchvision.models import mobilenet_v2
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize)

from raceai.utils.misc import race_load_class, race_report_result
from raceai.utils.logger import (race_set_loglevel, race_set_logfile, Logger)

import time # noqa
import zmq

view_debug = False
race_set_loglevel('info')
race_set_logfile('/tmp/raceai-yolo.log')

context = zmq.Context()
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))


def sort_contours(cnts, reverse = False):
    bboxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bboxes) = zip(*sorted(zip(cnts, bboxes),
        key=lambda b: b[1][0], reverse=reverse))
    return cnts


def detect_and_recognize(opt):
    Logger.info('loading weight [%s]...' % opt.det_weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    t0 = time.time()
    detnet = attempt_load(opt.det_weights, map_location=device)
    imgsz = check_img_size(opt.img_size, s=detnet.stride.max())
    if half:
        detnet.half()

    num_classes = 65
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5,0.5))])
    clsnet = mobilenet_v2(pretrained=False)
    clsnet.classifier[1] = torch.nn.Linear(clsnet.classifier[1].in_features, num_classes)
    clsnet.to(device);
    Logger.info('loading weight [%s]...' % opt.cls_weights)
    clsnet.load_state_dict(torch.load(opt.cls_weights))

    t1 = time.time()
    Logger.info('loading finish [%.2fs]' % (t1 - t0))
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    detnet(img.half() if half else img)
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
                ## Step-1
                dataset = LoadImages(image_path, img_size=imgsz)
                path, img, im0, _ = next(iter(dataset))
                # Logger.info(path)
                resitem = {'image_path': source, 'predict_box':[]}
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = detnet(img, augment=opt.augment)[0]
                if 'nms' in cfg:
                    if 'conf_thres' in cfg.nms:
                        conf_thres = cfg.nms.conf_thres
                    if 'iou_thres' in cfg.nms:
                        iou_thres = cfg.nms.iou_thres
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                if len(pred) == 0:
                    continue

                det = pred[0]
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                *xyxy, _conf, _cls = reversed(det)[0] # only best
                x1, y1, x2, y2 = [int(x.cpu()) for x in xyxy]
                img_rgb = im0[y1:y2, x1:x2, ::-1]

                ## Step-2
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                hsv_planes = cv2.split(hsv)
                hsv_planes[2] = clahe.apply(hsv_planes[2])
                hsv = cv2.merge(hsv_planes)
                img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                e = 3
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                img_rgb = cv2.copyMakeBorder(img_rgb, e, e, e, e, cv2.BORDER_CONSTANT, value=0) 
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                img_blur = cv2.medianBlur(img_gray, 3) 
                img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                img_morph = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

                conts = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if len(conts) < 7:
                    continue

                crop_characters = []
                for c in sort_contours(conts):
                    (x, y, w, h) = cv2.boundingRect(c)
                    ratio = h/w
                    if 1 <= ratio <= 3.5:
                        if h/img_rgb.shape[0] >= 0.3:
                            _img = cv2.resize(img_bin[y-e:y+h+e, x-e:x+w+e], dsize=(20, 20))
                            _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
                            crop_characters.append(transform(Image.fromarray(_img)))

                ## Step-3
                if len(crop_characters) < 7:
                    continue
                tensor_imgs = torch.stack(crop_characters, dim=0)
                clsnet.eval()
                with torch.no_grad():
                    output = clsnet(tensor_imgs.to(device))
                    result = output.argmax(dim=1).cpu().tolist()
                    resdata['result'].append(result)
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
    Logger.info('start ccpd2019 main')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_weights', type=str, default='/ckpts/cls.pt', help='detech model weight path(s)')
    parser.add_argument('--det_weights', type=str, default='/ckpts/det.pt', help='classifer model weight path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--topic', default='zmq.ccpd2019.inference', help='sub topic')
    opt = parser.parse_args()
    Logger.info(opt)

    zmqsub.subscribe(opt.topic)

    with torch.no_grad():
        Logger.info('start yolo detect')
        detect_and_recognize(opt)
        Logger.info('never run here')
