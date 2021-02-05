#!/bin/bash

SS_MODULE=${SS_MODULE:-"app"}
REDIS_ADDR=${REDIS_ADDR:-"0.0.0.0"}
REDIS_PORT=${REDIS_PORT:-"10090"}
REDIS_PSWD=${REDIS_PSWD:-"123456"}

while getopts "s:" OPT;
do
    case $OPT in
        s)
            SS_MODULE=$OPTARG
            ;;
        *)
            echo $OPTARG
            ;;
    esac
done

cd /raceai/codes/app

while true;
do
    case $SS_MODULE in
        "app")
            python app_service.py --host 0.0.0.0 --port 9119 --debug 0 \
            --redis_addr ${REDIS_ADDR} \
            --redis_port ${REDIS_PORT} \
            --redis_passwd ${REDIS_PSWD}
            ;;
        "yolov5")
            python zmq/yolo_inference.py \
                --weights /ckpts/${MODEL_LEVEL:-"x"}.pt \
                --topic zmq.yolov5${MODEL_LEVEL:-"x"}.inference \
                --img-size 640 --conf-thres 0.25 --iou-thres 0.85 \
                --device 0
            ;;
        *)
            echo "SS_MODULE set error"
            break;;
    esac
    sleep 3
done
