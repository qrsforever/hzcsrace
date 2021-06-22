#!/bin/bash
#=================================================================
# date: 2021-04-23 16:54:13
# title: start_alphapose
# author: QRS
#=================================================================


CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai_sense
REPOSITORY="$VENDOR/$PROJECT"
NAME=${PROJECT}

minio_server_url='s3-internal.didiyunapi.com'
minio_access_key='AKDD002E38WR1J7RMPTGRIGNVCVINY'
minio_secret_key='ASDDXYWs45ov7MNJbj5Wc2PM9gC0FSqCIkiyQkVC'

cmd="-s sense"
arg=""

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${NAME} \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env MINIO_SERVER_URL=$minio_server_url \
        --env MINIO_ACCESS_KEY=$minio_access_key \
        --env MINIO_SECRET_KEY=$minio_secret_key \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/sense:/ckpts \
        --volume /raceai/data/ckpts/sense/resources:/raceai/codes/projects/sense/resources \
        --volume /raceai/data/users/outputs:/outputs \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/app/configs/sense:/configs \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume $TOP_DIR/projects/sense/sense:/raceai/codes/projects/sense/sense \
        $REPOSITORY $cmd
}

if [[ x$1 == x1 ]]
then
    docker container stop ${NAME} > /dev/null
    docker container rm ${NAME} > /dev/null
fi

__start_raceai
