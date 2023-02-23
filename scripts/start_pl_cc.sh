#!/bin/bash
#=================================================================
# date: 2021-03-03 20:34:07
# title: start_pl_cr
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
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-pl.cc \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --restart unless-stopped \
        --network host \
        --env MODEL_WEIGHTS=best.ckpt \
        --env NUM_CLASSES=5 \
        --env TOPIC=zmq.cc.resnet18.inference \
        --env PRI_HTTP_PROXY=${HTTP_PROXY} \
        --volume /data/k12-nfs/raceai/data:/raceai/data \
        --volume /data/k12-nfs/raceai/data/ckpts/cartoon_characters/checkpoints:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        $REPOSITORY -s pl
}

__start_raceai
