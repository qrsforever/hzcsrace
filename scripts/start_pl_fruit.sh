#!/bin/bash
#=================================================================
# date: 2021-06-22 17:25:23
# title: start_pl_fruit
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

source ${TOP_DIR}/_env

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-pl.fruit \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --restart unless-stopped \
        --env MODEL_WEIGHTS=best.ckpt \
        --env NUM_CLASSES=5 \
        --env TOPIC=zmq.fruit.resnet18.inference \
        --env PRI_HTTP_PROXY=${HTTP_PROXY} \
        --volume ${DATA_ROOT}/raceai/data:/raceai/data \
        --volume ${DATA_ROOT}/raceai/data/ckpts/fruit/checkpoints:/ckpts \
        --volume ${TOP_DIR}/app:/raceai/codes/app \
        --volume ${TOP_DIR}/entrypoint.sh:/entrypoint.sh \
        $REPOSITORY -s pl
}

__start_raceai
