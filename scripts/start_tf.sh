#!/bin/bash
#=================================================================
# date: 2021-05-19 15:21:20
# title: start_tf
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai_tf
REPOSITORY="$VENDOR/$PROJECT"

__start_raceai()
{
    docker run -dit --runtime nvidia --name ${PROJECT}-repnet_tf \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env TOPIC=zmq.repnet_tf.inference \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/repnet_tf:/ckpts \
        --volume /raceai/data/users/outputs:/outputs \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume $TOP_DIR/projects/repnet-cli/repnet:/raceai/codes/projects/repnet-cli/repnet \
        --volume $TOP_DIR/projects/repnet-cli/utils:/raceai/codes/projects/repnet-cli/utils \
        $REPOSITORY
}

__start_raceai
