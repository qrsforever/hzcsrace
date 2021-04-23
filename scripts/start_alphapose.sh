#!/bin/bash
#=================================================================
# date: 2021-04-23 16:54:13
# title: start_alphapose
# author: QRS
#=================================================================


CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai_pose
REPOSITORY="$VENDOR/$PROJECT"

arg=""

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-alphapose \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/alphapose:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/projects/AlphaPose/configs:/raceai/codes/projects/AlphaPose/configs \
        --volume $TOP_DIR/projects/AlphaPose/scripts:/raceai/codes/projects/AlphaPose/scripts \
        --volume $TOP_DIR/projects/AlphaPose/alphapose:/raceai/codes/projects/AlphaPose/alphapose \
        --volume $TOP_DIR/projects/AlphaPose/detector:/raceai/codes/projects/AlphaPose/detector \
        $REPOSITORY 
}

while getopts "dm:" OPT;
do
    case $OPT in
        d)
            arg="it --restart unless-stopped"
            ;;
        *)
            echo "arg error"
            exit 0
    esac
done

__start_raceai
