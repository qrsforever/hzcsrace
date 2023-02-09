#!/bin/bash
#=================================================================
# date: 2021-03-03 20:44:24
# title: start_pl_garbage
# author: QRS
#=================================================================


CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

source ${TOP_DIR}/env.sh

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-pl.garbage \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env MODEL_WEIGHTS=pl_resnet18_acc85.pth \
        --env NUM_CLASSES=4 \
        --env TOPIC=zmq.garbage.resnet18.inference \
        --env PRI_HTTP_PROXY=${HTTP_PROXY} \
        --volume /data/k12-nfs/raceai/data:/raceai/data \
        --volume /data/k12-nfs/raceai/data/ckpts/rgarbage:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        $REPOSITORY -s pl
}

__start_raceai
