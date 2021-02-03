#!/bin/bash
#=================================================================
# date: 2021-02-03 15:42:11
# title: start_app
# author: QRS
#=================================================================


CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

REDIS_ADDR=${REDIS_ADDR:-'10.255.0.41'}
REDIS_PORT=10090
REDIS_PSWD='qY3Zh4xLPZNMkaz3'

cmd=""
arg=""

if [[ x$1 == xdev ]]
then
    arg="it --restart unless-stopped"
    shift
else
    cmd="-s app"
fi

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-app \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env REDIS_ADDR=${REDIS_ADDR} \
        --env REDIS_PORT=${REDIS_PORT} \
        --env REDIS_PSWD=${REDIS_PSWD} \
        --volume /raceai/data:/raceai/data \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        $REPOSITORY $cmd
}
__start_raceai
