#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file preprocessing.py
# @brief
# @author QRS
# @version 1.0
# @date 2021-03-16 16:49


import os
import shutil
import json
import pandas as pd


def convert_annotation(phase, in_dir, out_dir, sample_ids):
    in_images_path = f'{in_dir}/{phase}2019'
    out_images_path = f'{out_dir}/images/{phase}'
    out_labels_path = f'{out_dir}/labels/{phase}'

    os.makedirs(out_images_path, exist_ok=True)
    os.makedirs(out_labels_path, exist_ok=True)

    # 1. load json desc
    with open(os.path.join(f'{in_dir}/instances_{phase}2019.json'), 'rb') as f:
        data = json.load(f)

    imgs_df = pd.DataFrame(data['images'])
    anns_df = pd.DataFrame(data['annotations'])

    # 2. filter category id
    anns_df = anns_df[anns_df['category_id'].isin(sample_ids)]

    # 3. filter image id
    img_ids = anns_df['image_id'].unique()
    imgs_df = imgs_df[imgs_df['id'].isin(img_ids)]

    # 4. covert yolov5 format
    for _, item in imgs_df.iterrows():
        imgw, imgh = item.width, item.height
        # 4.1 normalize scale
        dw, dh = 1.0 / imgw, 1.0 / imgh
        img_src_path = os.path.join(in_images_path, item.file_name)
        if not os.path.exists(img_src_path):
            continue
        img_dst_path = os.path.join(out_images_path, item.file_name)
        lab_dst_path = os.path.join(out_labels_path, item.file_name.replace('.jpg', '.txt')) # noqa
        # 4.2 annotation in this image
        anns = anns_df[anns_df['image_id'] == item.id]
        labs = []
        for _, ann in anns.iterrows():
            # 4.3 convert bbox
            cls_id = sample_ids.index(ann.category_id)
            cx, cy = dw * ann.point_xy[0], dh * ann.point_xy[1]
            sw, sh = dw * ann.bbox[2], dh * ann.bbox[3]
            labs.append('%d %.6f %.6f %.6f %.6f' % (cls_id, cx, cy, sw, sh))
        # 4.4 save to file
        with open(lab_dst_path, 'w') as fw:
            fw.write('\n'.join(labs))
        # 4.5 copy image to out dir
        shutil.copyfile(img_src_path, img_dst_path)


sample_ids = [34, 71, 74, 77, 78, 83]

convert_annotation('train', '.', './out', sample_ids)
convert_annotation('val', '.', './out', sample_ids)
convert_annotation('test', '.', './out', sample_ids)
