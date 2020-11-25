#!/bin/bash

python3 /raceai/codes/projects/yolov5/detect.py \
    --source /raceai/data/tmp/det_test.jpeg \
    --weights /raceai/data/tmp/licenseplates_l/weights/best.pt \
    --project /raceai/data/tmp --name licenseplates_l --exist-ok \
    --conf 0.3 --device cpu
