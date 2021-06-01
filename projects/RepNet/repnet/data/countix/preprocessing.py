#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file preprocessing.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-05-12 14:11


import os
import cv2
import pandas as pd
import subprocess
import youtube_dl

YOUTUBE_PREFIX = 'https://www.youtube.com/watch?v='
DATASET_PREFIX = '/data/datasets/cv/countix'

FRAME_WIDTH, FRAME_HEIGHT = 112, 112
NUM_FRAMES = 64
NUM_DMODEL = 512
REP_OUT_TIME_RATE = 0.12

SOCKS5_PROXY = 'socks5://127.0.0.1:1881'

YDL_OPTS = {
    'format': 'mp4',
    'proxy': SOCKS5_PROXY,
    'quiet': True,
    'max_filesize': 30000000, # 30MB
}


def video_download_crop(vid, fps, wh, ss, to, raw_dir, out_dir, force=False):
    raw_file = f'{raw_dir}/{vid}.mp4'
    out_file = '%s/%s_%010.6f_%010.6f.mp4' % (out_dir, vid, ss, to)

    if os.path.exists(out_file):
        if force:
            os.remove(out_file)
        return out_file

    if not os.path.exists(raw_file):
        YDL_OPTS['outtmpl'] = raw_file
        with youtube_dl.YoutubeDL(YDL_OPTS) as ydl:
            ydl.download([f'{YOUTUBE_PREFIX}{vid}'])

    if os.path.exists(raw_file):
        cmd = 'ffmpeg -i %s -v 0 -r %f -s %s -ss %s -to %s -an %s' % (
                raw_file, fps, wh, ss, to, out_file)
        subprocess.call(cmd, shell=True)
        return out_file

    return None


def data_preprocess(data_prefix, phase, force=False):
    df = pd.read_csv(f'{data_prefix}/countix_{phase}.csv')
    raw_dir = f'{data_prefix}/raw/{phase}'
    out_dir = f'{data_prefix}/{phase}'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    df['file_name'] = None
    df['rep_start_frame'] = 0
    df['rep_end_frame'] = 0
    for idx, row in df.iterrows():
        if phase == 'test' or phase == 'sample':
            vid, ks, ke, rs, re, count, file_name, rsf, rse = row
        else:
            vid, _, ks, ke, rs, re, count, file_name, rsf, rse = row

        interval = re - rs
        cs = float(max([ks, rs - REP_OUT_TIME_RATE * interval]))
        ce = float(min([ke, re + REP_OUT_TIME_RATE * interval]))
        try:
            fps = NUM_FRAMES / (ce - cs)
            out_file = video_download_crop(vid,
                    fps, '%dx%d' % (FRAME_WIDTH, FRAME_HEIGHT), cs, ce, raw_dir, out_dir, force)
            if out_file is not None:
                cap = cv2.VideoCapture(out_file)
                cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if cnt >= NUM_FRAMES:
                    print('preprocess file: %s' % out_file)
                    df.loc[idx, 'rep_start_frame'] = int(fps * (rs - cs))
                    df.loc[idx, 'rep_end_frame'] = int(fps * (re - cs))
                    df.loc[idx, 'file_name'] = os.path.basename(out_file)
                else:
                    print(f'frames is less than {NUM_FRAMES}')
            else:
                print('download or crop [%s] fail' % vid)
        except Exception as err:
            print('%s' % err)
    sub_df = df[df['file_name'].notnull()]
    sub_df.to_csv(f'{data_prefix}/sub_countix_{phase}.csv', index=False, header=True)
    return sub_df

# data_preprocess(DATASET_PREFIX, 'test')
# data_preprocess(DATASET_PREFIX, 'val')
# data_preprocess(DATASET_PREFIX, 'train')
