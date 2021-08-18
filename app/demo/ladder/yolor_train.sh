#!/bin/bash

pro=ladder

# rm -rf /raceai/data/tmp/${pro}_r

python3 /raceai/codes/projects/yolor/train.py \
    --img-size 640 --batch-size 40 --epochs 820  --device 0 \
    --workers 4 --project /raceai/data/tmp --name ${pro}_r --exist-ok \
    --data dataset.yaml --cfg yolor_p6.cfg --hyp ./hyp.scratch.640.yaml \
    --weights /raceai/data/tmp/${pro}_r/weights/last.pt
#   --weights /raceai/data/tmp/yolor.pt
#   --weights /raceai/data/ckpts/yolor/yolor_p6.pt
