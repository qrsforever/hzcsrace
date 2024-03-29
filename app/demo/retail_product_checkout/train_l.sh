#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 32 --epochs 40  --device 0 \
    --workers 4 --project /raceai/data/tmp --name rpc_l --exist-ok \
    --data dataset.yaml --cfg yolov5l.yaml --hyp hyperparameters.yaml \
    --weights '/raceai/data/ckpts/yolov5/yolov5l.pt'
