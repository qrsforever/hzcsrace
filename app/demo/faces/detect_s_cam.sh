#!/bin/bash

python3 /raceai/codes/projects/yolov5/detect.py \
    --source 0 \
    --weights /raceai/data/tmp/faces_s/weights/best.pt \
    --project /raceai/data/tmp --name faces_s --exist-ok \
    --conf-thres 0.3 --iou-thres 0.6 --device cpu
