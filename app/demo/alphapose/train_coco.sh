set -x

python ./train.py \
    --work_dir /raceai/data/tmp/alphapose/output/coco \
    --cfg coco/256x192_res50_lr1e-3_1x.yaml
