#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 48 --epochs 100  --device 0 \
    --workers 4 --project /raceai/data/tmp --name rpc_m --exist-ok \
    --data dataset.yaml --cfg yolov5m.yaml --hyp hyperparameters.yaml \
    --weights /raceai/data/tmp/rpc_m/weights/last.pt
