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
cmd="-s yolov5"
task="rd"
model="l" # [s,l,x]

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-yolov5-$task-$model \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env TASK=$task \
        --env MODEL_LEVEL=$model \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/yolov5/road_damage:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume /data/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/projects/yolov5:/raceai/codes/projects/yolov5 \
        $REPOSITORY $cmd
}

while getopts "dm:" OPT;
do
    case $OPT in
        d)
            arg="it --restart unless-stopped"
            cmd=""
            ;;
        m)
            model=$OPTARG
            ;;
        *)
            echo "arg error"
            exit 0
    esac
done

__start_raceai