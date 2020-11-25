#!/bin/bash

python3 /raceai/codes/projects/yolov5/detect.py \
    --source /raceai/data/tmp/det_test.jpeg \
    --weights /raceai/data/tmp/licenseplates_s/weights/best.pt \
    --project /raceai/data/tmp --name licenseplates_s --exist-ok \
    --conf-thres 0.4 --iou-thres 0.8 --device cpu
