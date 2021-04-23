set -x

python ./train.py \
    --work_dir /raceai/data/tmp/alphapose/output/halpe \
    --cfg halpe_136/256x192_res50_lr1e.yaml
