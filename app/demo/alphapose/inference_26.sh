#!/bin/bash

set -x

CONFIG=halpe_26/256x192_res50_lr1e-3_1x.yaml
CKPT=/raceai/data/ckpts/alphapose/halpe26_fast_res50_256x192.pth
VIDEO=/raceai/data/media/videos/alphapose_test.mp4
OUTDIR=/raceai/data/tmp/alphapose/output/halpe26

python ./demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --vis_fast \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
