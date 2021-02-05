#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 16 --epochs 30  --device 0 \
    --workers 4 --project /raceai/data/tmp --name faces_x --exist-ok \
    --data dataset.yaml --cfg yolov5x.yaml --hyp hyperparameters.yaml \
    --weights '/raceai/data/ckpts/yolov5/yolov5x.pt'
