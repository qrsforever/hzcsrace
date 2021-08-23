#!/bin/bash

pro=ladder
out=/raceai/data/tmp

# rm -rf /raceai/data/tmp/${pro}_l

python3 /raceai/codes/projects/yolov5/train.py \
    --img-size 640 --batch-size 20 --epochs 320  --device 0 \
    --workers 4 --project $out --name ${pro}_l --exist-ok \
    --data ./dataset.yaml \
    --hyp ./hyperparameters.yaml \
    --cfg ./yolov5l_small_achor.yaml \
    --weights /raceai/data/ckpts/yolov5/yolov5l.pt
#   --weights ""
#   --weights $out/${pro}_l/weights/last.pt
#   --weights /raceai/data/ckpts/yolov5/ladder/l.pt
