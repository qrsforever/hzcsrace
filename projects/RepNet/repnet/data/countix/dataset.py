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
import glob
import os.path as osp
import random
from random import randint
import torch.nn.functional as F

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


class CountixSynthDataset(Dataset):
    def __init__(self, data_root, phase, frame_size=112, num_frames=64):
        self.data_root = data_root
        self.num_frames = num_frames
        self.frame_size = (frame_size, frame_size) if isinstance(frame_size, int) else frame_size
        self.data = self._make_valid_data(data_root, phase)

    def _make_valid_data(self, data_root, phase):
        synthvids_paths = glob.glob(f'{data_root}/synthvids/*.mp4')
        df_all = pd.read_csv(f'{data_root}/countix/countix_{phase}.csv')
        valid_index = []
        valid_paths = []
        for i in range(len(df_all)):
            vpath = f'{data_root}/{phase}vids/{phase}{i}.mp4'
            count = df_all.loc[i, 'count'] 
            if count > 32 or count < 2:
                # print(vpath)
                continue
            if osp.exists(vpath):
                valid_index.append(i)
                valid_paths.append(vpath)
        df_valid = df_all.iloc[valid_index]
        valid_counts = df_valid['count']
        random_synth = random.choices(synthvids_paths, k=len(valid_paths))
        return list(zip(valid_paths, valid_counts, random_synth))

    def _get_frames(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            img = Image.fromarray(frame)
            frames.append(img)

        cap.release()
        assert len(frames) > 0, path
        return frames

    def _skip_frames(self, frames, count):
        new_frames = []
        f_len = len(frames)
        for i in range(1, count + 1):
            new_frames.append(frames[i * f_len // count - 1])
        return new_frames

    def __getitem__(self, index):
        cvid_path, cvid_count, svid_path = self.data[index]
        cframes = self._get_frames(cvid_path)
        sframes = self._get_frames(svid_path)

        cvid_len = min(len(cframes), randint(
            max(2 * cvid_count, int(0.7 * self.num_frames)), self.num_frames))
        head_len = randint(0, self.num_frames - cvid_len)
        tail_len = self.num_frames - cvid_len - head_len

        cframes = self._skip_frames(cframes, cvid_len)
        sframes = self._skip_frames(sframes, head_len + tail_len)

        same = np.random.choice([0, 1], p=[0.5, 0.5])
        if same:
            final_frames = [cframes[0] for i in range(head_len)]
            final_frames.extend(cframes)
            final_frames.extend([cframes[-1] for i in range(tail_len)])
        else:
            final_frames = sframes[:head_len]
            final_frames.extend(cframes)
            final_frames.extend(sframes[head_len:])

        Xlist = []
        for img in final_frames:
            preprocess = T.Compose([
                T.Resize(self.frame_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            Xlist.append(preprocess(img).unsqueeze(0))

        p_len = cvid_len / cvid_count

        assert 2 <= p_len <= 32, "cvid_len: %d, cvid_count: %d" % (cvid_len, cvid_count)

        y1 = np.full((self.num_frames, 1), fill_value=p_len)
        y2 = np.ones((self.num_frames, 1))

        for i in range(self.num_frames):
            if i < head_len or i > (self.num_frames - tail_len):
                y1[i] = 0
                y2[i] = 0
                Xlist[i] = F.dropout(Xlist[i], 0.2)

        X = torch.cat(Xlist)
        y1 = torch.FloatTensor(y1)
        y2 = torch.FloatTensor(y2)
        y3 = torch.FloatTensor([cvid_count])

        return X, y1, y2, y3

    def __len__(self):
        return len(self.data)
