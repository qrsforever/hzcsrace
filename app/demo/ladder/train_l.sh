#!/bin/bash

pro=ladder

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 12 --epochs 100  --device 0 \
    --workers 4 --project /raceai/data/tmp --name ${pro}_l --exist-ok \
    --data dataset.yaml --cfg yolov5l.yaml --hyp hyperparameters.yaml \
    --weights '/raceai/data/ckpts/yolov5/yolov5l.pt'
