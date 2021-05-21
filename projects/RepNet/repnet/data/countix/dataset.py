#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file dataset.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-12 14:13


import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from torchvision import transforms as T
from torch.utils.data import Dataset


class CountixDataset(Dataset):

    def __init__(self, data_root, phase, frame_size=112, num_frames=64):
        self.data_root = data_root
        self.phase = phase
        self.num_frames = num_frames
        self.frame_size = (frame_size, frame_size) if isinstance(frame_size, int) else frame_size

        self.df = pd.read_csv(f'{data_root}/sub_countix_{phase}.csv')

    def __getitem__(self, index):
        item = self.df.iloc[index]
        start = item.rep_start_frame
        end = item.rep_end_frame
        count = self.df.loc[index, 'count']

        path = f'{self.data_root}/{self.phase}/{item.file_name}'

        frames = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            img = Image.fromarray(frame)
            trans = T.Compose([
                T.Resize(self.frame_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
            frames.append(trans(img).unsqueeze(0))
        cap.release()

        X = frames[:self.num_frames]
        X = torch.cat(X)

        period_length = (end - start) / count

        y1 = np.full((self.num_frames, 1), fill_value=period_length)
        y2 = np.ones((self.num_frames, 1))
        for i in range(self.num_frames):
            if i < start or i > end:
                y1[i] = 0
                y2[i] = 0

        # y1 = torch.LongTensor(y1)
        y1 = torch.FloatTensor(y1)
        y2 = torch.FloatTensor(y2)
        y3 = torch.FloatTensor([count])
        return X, y1, y2, y3

    def __len__(self):
        return len(self.df)
