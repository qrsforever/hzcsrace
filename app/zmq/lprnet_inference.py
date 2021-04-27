#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file lprnet_inference.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-04-15 20:04


import traceback
import argparse
import cv2
import torch
import numpy as np
from torch import nn
from PIL import Image
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
zmqsub = context.socket(zmq.SUB)
zmqsub.connect('tcp://{}:{}'.format('0.0.0.0', 5555))

CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS)
BLANK_IDX = NUM_CLASS - 1
INPUT_SIZE = (94, 24) # (w, h)

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        ch = ch_out // 4
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, num_class, dropout_rate = 0.5):
        super(LPRNet, self).__init__()
        self.num_class = num_class
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64), # 1
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)), # 3
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128), # 5
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)), # 7
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256), # 9
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(), # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate), # 15
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256), # 17
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate), # 19
            nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=num_class), # 21
            nn.ReLU(),  # *** 22 ***
        )
        # 448 = 64 + 128 + 256
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.num_class,
                      out_channels=self.num_class,
                      kernel_size=(1, 1), stride=(1, 1)),
        )

        # init weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)
        x = torch.cat(global_context, 1)
        x = self.container(x)  # (batch, n_class, 4, timestep)
        logits = torch.mean(x, dim=2) # (batch, n_class, timestep)
        return logits


def detect_and_recognize(opt):
    Logger.info('loading weight [%s]...' % opt.det_weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    t0 = time.time()
    detnet = attempt_load(opt.det_weights, map_location=device)
    imgsz = check_img_size(opt.img_size, s=detnet.stride.max())
    if half:
        detnet.half()

    model = LPRNet(num_class=NUM_CLASS).to(device)
    model.load_state_dict(torch.load(opt.cls_weights))
    model.eval()

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
                img_bgr = im0[y1:y2, x1:x2, :]
                img_bgr = cv2.resize(img_bgr, INPUT_SIZE).astype(np.float32)
                img_bgr -= 127.5
                img_bgr *= 0.0078125  # 1/128
                img_bgr = np.transpose(img_bgr, (2, 0, 1)) # (C, H, W)
                img_tensor = torch.from_numpy(img_bgr)

                with torch.no_grad():
                    inputs = img_tensor.unsqueeze(dim=0)
                    preds = model(inputs.to(device))
                    preds = preds.cpu().detach()
                    labels_with_blank = preds[0, :, :].argmax(dim=0).numpy()
                    labels = []
                    check_repeat = False
                    for x in labels_with_blank:
                        if x == BLANK_IDX:
                            check_repeat = False
                            continue
                        if check_repeat and labels[-1] == x:
                            continue
                        labels.append(int(x))
                        check_repeat = True
                    label_names = [CHARS[j] for j in labels]
                    resdata['result'].append({"label_ids": labels, "label_names": label_names})

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
    parser.add_argument('--det_weights', type=str, default='/ckpts/det.pt', help='detach model weight path(s)')
    parser.add_argument('--cls_weights', type=str, default='/ckpts/lprnet.pt', help='classifer model weight path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--topic', default='zmq.lprnet.inference', help='sub topic')
    opt = parser.parse_args()
    Logger.info(opt)

    zmqsub.subscribe(opt.topic)
    race_report_result('add_topic', opt.topic)

    try:
        with torch.no_grad():
            Logger.info('start yolo detect')
            detect_and_recognize(opt)
            Logger.info('never run here')
    finally:
        race_report_result('del_topic', opt.topic)
