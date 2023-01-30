#!/bin/bash
#=================================================================
# date: 2021-08-17
# title: start_yolor
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai
REPOSITORY="$VENDOR/$PROJECT"

arg=""
cmd="-s yolor"
task="ladder"

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-yolor-$task \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env TASK=$task \
        --volume /data/k12-nfs/raceai/data:/raceai/data \
        --volume /data/k12-nfs/raceai/data/ckpts/yolor/$task:/ckpts \
        --volume /data/k12-nfs/pretrained/cv:/root/.cache/torch/hub/checkpoints \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume $TOP_DIR/projects/yolor:/raceai/codes/projects/yolor \
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
