#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 64 --epochs 100  --device 0 \
    --workers 4 --project /raceai/data/tmp --name rpc_s --exist-ok \
    --data dataset.yaml --cfg yolov5s.yaml --hyp hyperparameters.yaml \
    --weights '/raceai/data/ckpts/yolov5/yolov5s.pt'
