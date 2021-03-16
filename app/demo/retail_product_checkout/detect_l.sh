#!/bin/bash

python3 /raceai/codes/projects/yolov5/detect.py \
    --source /raceai/data/tmp/det_test.jpeg \
    --weights /raceai/data/tmp/rpc_l/weights/best.pt \
    --project /raceai/data/tmp --name rpc_l --exist-ok \
    --conf-thres 0.4 --iou-thres 0.8 --device cpu
