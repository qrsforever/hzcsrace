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
else
    cmd="python3 app_service.py"
fi

docker run -d${arg} --runtime nvidia --name ${PROJECT} \
    --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host --entrypoint /bin/bash \
    --volume /raceai/data:/raceai/data \
    --volume $TOP_DIR/app:/raceai/app \
    $REPOSITORY $cmd
