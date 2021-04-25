set -x

master_addr=10.255.0.58
master_port=8555
nnodes=5

python3 -m torch.distributed.launch \
        --nproc_per_node=4 --nnodes=$nnodes --node=$1 \
        --master_addr $master_addr --master_port $master_port \
        train.py --snapshot 3 \
	--work_dir /raceai/data/tmp/alphapose/output/halpe \
	--cfg halpe_136/256x192_res50_lr1e-3_2x-regression.yaml

# python ./train.py \
#     --snapshot 10 \
#     --work_dir /raceai/data/tmp/alphapose/output/halpe \
#     --cfg halpe_136/256x192_res50_lr1e-3_2x-regression.yaml
