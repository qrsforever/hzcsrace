#!/bin/bash

python3 /raceai/codes/projects/yolov5/train_s.py \
    --img-size 640 --batch-size 8 --epochs 50  --device 0 \
    --workers 4 --project /raceai/tmp --name licenseplates_s --exist-ok \
    --data dataset.yaml --cfg yolov5s.yaml --hyp hyperparameters.yaml \
    --weights '/raceai/data/ckpts/yolov5/yolov5s.pt'
