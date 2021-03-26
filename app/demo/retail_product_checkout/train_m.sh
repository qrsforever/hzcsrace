#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 32 --epochs 30  --device 0 \
    --workers 6 --project /raceai/data/tmp --name rpc_m --exist-ok \
    --data dataset.yaml --cfg yolov5m.yaml --hyp hyperparameters.yaml \
    --weights '/raceai/data/ckpts/yolov5/yolov5m.pt'
