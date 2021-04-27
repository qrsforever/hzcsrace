set -x

master_addr=10.255.0.58
master_port=8555
nnodes=5

pids=`ps -eo pid,args | grep "train.py" | grep -v grep | cut -c1-5`

echo $pids

if [[ x$pids != x ]]
then
    kill -9 $pids
fi

if [[ x$1 != x ]]
then
    python3 -m torch.distributed.launch \
            --nproc_per_node=1 --nnodes=$nnodes --node_rank=$1 \
            --master_addr $master_addr --master_port $master_port \
            train.py --snapshot 10 --rank $1 \
            --work_dir /raceai/data/tmp/alphapose/output/halpe \
            --cfg halpe_136/256x192_res50_lr1e-3_2x-regression.yaml
else
    python ./train.py \
        --snapshot 10 \
        --work_dir /raceai/data/tmp/alphapose/output/halpe \
        --cfg halpe_136/256x192_res50_lr1e-3_2x-regression.yaml
