#!/bin/bash

png=test.png
pro=ladder
pro_dir=/raceai/data/tmp

if [[ x$1 != x ]]
then
    png=$1
fi

rm -rf ${pro_dir}/${pro}_l/labels
rm -rf ${pro_dir}/${pro}_l/*.png

python3 /raceai/codes/projects/yolov5/detect.py \
    --img-size 640 \
    --source ${pro_dir}/${png} --save-txt \
    --project ${pro_dir} --name ${pro}_l --exist-ok \
    --conf-thres 0.1 --iou-thres 0.1 --device cpu \
    --weights ${pro_dir}/${pro}_l/weights/last.pt
