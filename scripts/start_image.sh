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

docker run -d --runtime nvidia --name ${PROJECT} --restart unless-stopped \
    --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --volume /raceai/data:/raceai/data \
    --volume $TOP_DIR/app:/raceai/app \
    $REPOSITORY python3 app_service.py
