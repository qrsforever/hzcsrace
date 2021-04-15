#!/bin/bash
#=================================================================
# date: 2021-02-03 15:46:00
# title: start_yolo
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

arg=""

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-lprnet \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/ccpd2019:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume $TOP_DIR/projects/yolov5:/raceai/codes/projects/yolov5 \
        $REPOSITORY -s lprnet
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
