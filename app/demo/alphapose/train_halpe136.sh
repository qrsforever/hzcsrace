set -x

python ./train.py \
    --snapshot 10 \
    --work_dir /raceai/data/tmp/alphapose/output/halpe \
    --cfg halpe_136/256x192_res50_lr1e-3_2x-regression.yaml
