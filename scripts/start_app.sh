#!/bin/bash
#=================================================================
# date: 2021-02-03 15:42:11
# title: start_app
# author: QRS
#=================================================================


CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

source ${TOP_DIR}/_env

VENDOR=hzcsai_com
PROJECT=raceai_app
REPOSITORY="$VENDOR/$PROJECT"

REDIS_ADDR=${REDIS_ADDR}
REDIS_PORT=${REDIS_PORT}
REDIS_PSWD=${REDIS_PSWD}

cmd=""
arg=""

if [[ x$1 == xdev ]]
then
    arg="it --restart unless-stopped"
    shift
else
    cmd="-s app"
fi

HTTP_PROXY=${HTTP_PROXY:-''}

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-app \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host --restart unless-stopped \
        --env REDIS_ADDR=${REDIS_ADDR} \
        --env REDIS_PORT=${REDIS_PORT} \
        --env REDIS_PSWD=${REDIS_PSWD} \
        --env PRI_HTTP_PROXY=${HTTP_PROXY} \
        --volume ${DATA_ROOT}/raceai/data:/raceai/data \
        --volume ${DATA_ROOT}/raceai/data/users/outputs:/outputs \
        --volume ${DATA_ROOT}/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume ${TOP_DIR}/app:/raceai/codes/app \
        --volume ${TOP_DIR}/projects:/raceai/codes/projects \
        --volume ${TOP_DIR}/entrypoint.sh:/entrypoint.sh \
        $REPOSITORY $cmd
}
__start_raceai
