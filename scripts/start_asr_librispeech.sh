#!/bin/bash
#=================================================================
# date: 2021-03-25 15:28:54
# title: start_asr_librispeech
# author: QRS
#=================================================================

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com
PROJECT=raceai.sb
REPOSITORY="$VENDOR/$PROJECT"

arg=""
cmd="-s asr"
task="librispeech"
model="asr-crdnn-rnnlm-librispeech"

__start_raceai()
{
    docker run -d${arg} --runtime nvidia --name ${PROJECT}-${model} \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        --network host \
        --env TASK=$task \
        --env MODEL=$model \
        --env TOPIC=zmq.asr.${task}.inference \
        --volume /raceai/data:/raceai/data \
        --volume /raceai/data/ckpts/asr:/ckpts \
        --volume $TOP_DIR/app:/raceai/codes/app \
        --volume $TOP_DIR/entrypoint.sh:/entrypoint.sh \
        --volume $TOP_DIR/projects/speechbrain:/raceai/codes/projects/speechbrain \
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
