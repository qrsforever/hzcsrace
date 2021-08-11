#!/bin/bash

png=test.png
pro=ladder

if [[ x$1 != x ]]
then
    png=$1
fi

rm -rf /raceai/data/tmp/${pro}_l/labels
rm -rf /raceai/data/tmp/${pro}_l/*.png

python3 /raceai/codes/projects/yolov5/detect.py \
    --source /raceai/data/tmp/${png} --save-txt \
    --weights /raceai/data/tmp/${pro}_l/weights/best.pt \
    --project /raceai/data/tmp --name ${pro}_l --exist-ok \
    --conf-thres 0.8 --iou-thres 0.6 --device cpu
