#!/bin/bash

pro=ladder
out=/raceai/data/tmp

# rm -rf /raceai/data/tmp/${pro}_l

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 40 --epochs 220  --device 0 \
    --workers 4 --project $out --name ${pro}_l --exist-ok \
    --data dataset.yaml --cfg yolov5l.yaml --hyp hyperparameters.yaml \
    --weights $out/best.pt
#   --weights /raceai/data/ckpts/yolov5/ladder/l.pt
#   --weights $out/${pro}_l/weights/last.pt
