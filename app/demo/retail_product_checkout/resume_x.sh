#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 24 --epochs 40  --device 0 \
    --workers 4 --project /raceai/data/tmp --name rpc_x --exist-ok \
    --data dataset.yaml --cfg yolov5x.yaml --hyp hyperparameters.yaml \
    --weights /raceai/data/tmp/rpc_x/weights/last.pt
