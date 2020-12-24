#!/bin/bash
#=================================================================
# date: 2020-11-09
# title: build_images
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
    cmd="python3 /raceai/codes/app/app_service.py"
fi

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT} \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --volume /raceai/data:/raceai/data \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/projects/yolov5:/raceai/codes/projects/yolov5 \
        --volume $TOP_DIR/projects/fsgan:/raceai/codes/projects/fsgan \
        --volume $TOP_DIR/projects/face_detection_dsfd:/raceai/codes/projects/face_detection_dsfd \
        $REPOSITORY $cmd
}

__start_fsgan()
{
    docker run -dit --runtime nvidia --name fsgan \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/projects/fsgan:/raceai/codes/project/fsgan \
        --volume $TOP_DIR/projects/face_detection_dsfd:/raceai/codes/project/face_detection_dsfd \
        $REPOSITORY.fsgan
}

if [[ x$1 == xfsgan ]]
then
    __start_fsgan
else
    __start_raceai
fi
