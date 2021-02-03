#!/bin/bash
#=================================================================
# date: 2021-02-03 15:46:00
# title: start_yolo
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

cmd=""
arg=""

if [[ x$1 == xdev ]]
then
    arg="it --restart unless-stopped"
    shift
else
    cmd="-s yolo"
fi

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-yolo \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --volume /raceai/data:/raceai/data \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/projects/yolov5:/raceai/codes/projects/yolov5 \
        $REPOSITORY $cmd
}
__start_raceai
