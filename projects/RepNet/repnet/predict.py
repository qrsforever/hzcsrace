#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file predict.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-06-03 10:24


import os
import argparse
import torch
import cv2
from PIL import Image
from torchvision import transforms as T

# from repnet.models.repnet import RepNet
from repnet.models.repnet2 import RepNet


def run_predict(opt):
    device = torch.device("cuda")

    # frames
    frames = []
    cap = cv2.VideoCapture(opt.video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        img = Image.fromarray(frame)
        frames.append(img)
    cap.release()

    new_frames = []
    num_frames = 64
    f_len = len(frames)
    for i in range(1, num_frames + 1):
        new_frames.append(frames[i * f_len // num_frames - 1])
    Xlist = []
    for img in new_frames:
        preprocess = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        Xlist.append(preprocess(img).unsqueeze(0))
    X = torch.cat(Xlist).unsqueeze(0).to(device)
    print(X.shape)

    checkpoint = torch.load(opt.ckpt_from_path)
    model = RepNet(num_frames, 512)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        y1_pred, y2_pred = model(X)
        y3_pred = torch.round(torch.sum((y2_pred > 0) / (y1_pred + 1e-1), 1))
        print(y1_pred)
        print(torch.sigmoid(y2_pred))
        print(y3_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--video_path',
            type=str,
            dest='video_path',
            help="")
    parser.add_argument(
            '--ckpt_from_path',
            default='last.pt',
            type=str,
            dest='ckpt_from_path',
            help="")

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_from_path):
        raise 'no ckpt path'

    run_predict(args)
