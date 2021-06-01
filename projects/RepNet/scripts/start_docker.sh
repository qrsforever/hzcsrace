#!/bin/bash
#=================================================================
# date: 2021-02-03 15:42:11
# title: start_app
# author: QRS
#=================================================================


CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai_base
REPOSITORY="$VENDOR/$PROJECT"

REDIS_ADDR=${REDIS_ADDR:-'10.255.0.41'}
REDIS_PORT=10090
REDIS_PSWD='qY3Zh4xLPZNMkaz3'

cmd=""
arg="it --restart unless-stopped"


__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-test \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --entrypoint bash \
        --network host \
        --env REDIS_ADDR=${REDIS_ADDR} \
        --env REDIS_PORT=${REDIS_PORT} \
        --env REDIS_PSWD=${REDIS_PSWD} \
        --volume /data:/data \
        --volume /raceai/data:/raceai/data \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        $REPOSITORY $cmd
}
__start_raceai
