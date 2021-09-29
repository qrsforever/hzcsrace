#!/bin/bash

png=test.png
pro=ladder

if [[ x$1 != x ]]
then
    png=$1
fi

rm -rf /raceai/data/tmp/${pro}_r/labels
rm -rf /raceai/data/tmp/${pro}_r/*.png

python3 /raceai/codes/projects/yolor/detect.py \
    --img-size 640 --classes 0 \
    --source /raceai/data/tmp/${png} --names names.txt \
    --cfg yolor_p6.cfg --output /raceai/data/tmp/${pro}_out \
    --conf-thres 0.2 --iou-thres 0.15 --device 0 \
    --weights /raceai/data/tmp/${pro}_r/weights/last.pt 
