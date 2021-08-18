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
    --img-size 640 \
    --source /raceai/data/tmp/${png} --save-txt \
    --project /raceai/data/tmp --name ${pro}_l --exist-ok \
    --conf-thres 0.2 --iou-thres 0.15 --device cpu \
    --weights /raceai/data/tmp/best.pt

#   --weights /raceai/data/tmp/${pro}_l/weights/last.pt

mv /raceai/data/tmp/${pro}_l/${png} /raceai/data/tmp/result_${png}
