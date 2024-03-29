#!/bin/bash

SS_MODULE=${SS_MODULE:-"app"}

REDIS_ADDR=${REDIS_ADDR:-"172.21.0.2"}
REDIS_PORT=${REDIS_PORT:-"10090"}
REDIS_PSWD=${REDIS_PSWD:-"qY3Zh4xLPZNMkaz3"}

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
                --topic zmq.yolov5.$TASK.${MODEL_LEVEL:-"x"}.inference \
                --img-size 640 --conf-thres 0.25 --iou-thres 0.85 \
                --device 0
            ;;
        "yolor")
            sleep 100
            ;;
        "pl")
            python3 zmq/pl_inference.py \
                --weights /ckpts/${MODEL_WEIGHTS} \
                --num_classes ${NUM_CLASSES} \
                --topic ${TOPIC}
            ;;
        "asr")
            python3 zmq/asr_inference.py \
                --ckpts /ckpts/${MODEL} \
                --topic ${TOPIC}
            ;;
        "ccpd2019")
            python3 zmq/ccpd2019_inference.py \
                --det_weights /ckpts/det.pt \
                --cls_weights /ckpts/cls.pt \
                --topic zmq.ccpd2019.inference \
                --img-size 640 --conf-thres 0.25 --iou-thres 0.85 \
                --device 0
            ;;
        "lprnet")
            python3 zmq/lprnet_inference.py \
                --det_weights /ckpts/det.pt \
                --cls_weights /ckpts/lprnet.pt \
                --topic zmq.lprnet.inference \
                --img-size 640 --conf-thres 0.25 --iou-thres 0.85 \
                --device 0
            ;;
        "alphapose")
            python3 zmq/alphapose_inference.py \
            --cfg /configs/256x192_res50_lr1e-3_1x.yaml \
            --topic zmq.alphapose.inference \
            --checkpoint /ckpts/halpe26_fast_res50_256x192.pth \
            --outdir /outputs
            ;;
        "repnet_tf")
            python3 zmq/repnet_tf_inference.py \
                --topic zmq.repnet_tf.inference \
                --ckpt /ckpts \
                --out /outputs
            ;;
        "sense")
            python3 zmq/sense_inference.py \
            --topic zmq.sense.inference \
            --custom_classifier /ckpts \
            --outdir /outputs
            ;;
        "pcaks")
            python3 zmq/pcaks_test.py \
                --topic ${TOPIC}
            ;;
        *)
            echo "SS_MODULE set error"
            break;;
    esac
    sleep 3
done
