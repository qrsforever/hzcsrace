#!/bin/bash

set -x

# CONFIG=256x192_res50_lr1e-3_1x.yaml
CONFIG=halpe_136/256x192_res50_lr1e-3_2x-regression.yaml
# CKPT=/raceai/data/ckpts/alphapose/halpe136_fast_res50_256x192.pth
CKPT=/raceai/data/tmp/alphapose/output/halpe/256x192_res50_lr1e-3_2x-regression.yaml/model_4.pth
# IMAGE=/raceai/data/media/images/alphapose_test.jpg
VIDEO=/raceai/data/media/videos/alphapose_test.mp4
OUTDIR=/raceai/data/tmp/alphapose/output/halpe

# python ./demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --image ${IMAGE} \
#     --outdir ${OUTDIR} \
#     --detector yolo  --save_img --save_video

python ./demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --vis_fast \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
