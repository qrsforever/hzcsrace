#!/bin/bash

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 32 --epochs 200  --device 0 \
    --workers 4 --project /raceai/data/tmp --name road_damage_l --exist-ok \
    --data dataset.yaml --cfg yolov5l.yaml --hyp hyperparameters.yaml \
    --weights /raceai/data/tmp/road_damage_l/weights/last.pt
