#!/bin/bash
#=================================================================
# date: 2021-03-03 20:34:07
# title: start_pl_cr
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-pl.cr \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env MODEL_WEIGHTS=pl_resnet18_acc90.pth \
        --env NUM_CLASSES=3 \
        --env TOPIC=zmq.cr.resnet18.inference \
        --volume /root/raceai/data:/raceai/data \
        --volume /root/raceai/data/ckpts/cleaner_robot:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        $REPOSITORY -s pl
}

__start_raceai
