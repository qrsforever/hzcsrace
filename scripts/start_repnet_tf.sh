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

minio_server_url='s3-internal.didiyunapi.com'
minio_access_key='AKDD002E38WR1J7RMPTGRIGNVCVINY'
minio_secret_key='ASDDXYWs45ov7MNJbj5Wc2PM9gC0FSqCIkiyQkVC'

NAME=${PROJECT}-repnet_tf

__start_raceai()
{
    docker run -dit --runtime nvidia --name ${NAME} \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env TOPIC=zmq.repnet_tf.inference \
        --env PYTHONPATH=/raceai/codes/app:/raceai/codes/projects/repnet-cli \
        --env MINIO_SERVER_URL=$minio_server_url \
        --env MINIO_ACCESS_KEY=$minio_access_key \
        --env MINIO_SECRET_KEY=$minio_secret_key \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/repnet_tf:/ckpts \
        --volume /raceai/data/users/outputs:/outputs \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume $TOP_DIR/projects/repnet-cli:/raceai/codes/projects/repnet-cli \
        $REPOSITORY -s repnet_tf
}

if [[ x$1 == x1 ]]
then
    docker container stop ${NAME} > /dev/null
    docker container rm ${NAME} > /dev/null
fi

__start_raceai
