#!/bin/bash

set -x

CONFIG=coco/256x192_res50_lr1e-3_1x.yaml
# CONFIG=coco/256x192_res50_lr1e-3_2x-regression.yaml
CKPT=/raceai/data/ckpts/alphapose/fast_res50_256x192.pth
IMAGE=/raceai/data/media/images/alphapose_test.jpg
OUTDIR=/raceai/data/tmp/alphapose/output/coco

python ./demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --image ${IMAGE} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
