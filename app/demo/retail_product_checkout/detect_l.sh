#!/bin/bash

python3 /raceai/codes/projects/yolov5/detect.py \
    --source /raceai/data/tmp/6926475206263_camera0-39.jpg \
    --weights /raceai/data/tmp/rpc_l/weights/best.pt \
    --project /raceai/data/tmp --name rpc_l --exist-ok \
    --conf-thres 0.4 --iou-thres 0.8 --device cpu
