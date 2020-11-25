#!/bin/bash

python3 /raceai/codes/projects/yolov5/detect.py \
    --source 0 \
    --weights /raceai/data/tmp/licenseplates_s/weights/best.pt \
    --project /raceai/data/tmp --name licenseplates_s --exist-ok \
    --conf-thres 0.3 --iou-thres 0.6 --device cpu
